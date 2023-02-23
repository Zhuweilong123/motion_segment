# Copyright 2020 Toyota Research Institute.  All rights reserved.
# 主要任务：创建各个文件的路径（以字典、列表的格式）、读取深度信息、
# 这里的整个数据的读取流程为：（以raw dataset为例）编辑一个train_txt文件包含了训练集所有数据的路径和文件名→找到这些RGB图片对应的深度图→
import skimage
import torch.multiprocessing

import numpy as np
import os
import PIL.Image as pil

from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from dataset.original_method.kitti_dataset_utils import read_calib_file
from dataset.utils.image import load_image


from dataset.utils.transforms import transform_rgb, transform_depth

########################################################################################################################

# Cameras from the stero pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}

# Name of different calibration files
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}
PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'


########################################################################################################################
#### DATASET
########################################################################################################################
class KITTIDataset(Dataset):
    KITTI = namedtuple('KITTI', ['name', 'id', 'train_id', 'category', 'category_id', 'color'])
    # 调色板
    classes = [
        KITTI('unlabeled', 0, 0, 'void', 0, (0, 0, 0)),
        KITTI('car', 1, 1, 'vehicle', 1, (128, 0, 0)),
        KITTI('van', 2, 2, 'vehicle', 1, (0, 128, 0)),
        KITTI('truck', 3, 3, 'vehicle', 1, (128, 128, 0)),
        KITTI('pedestrian', 4, 4, 'human', 2, (0, 0, 128)),
        KITTI('sitter', 5, 5, 'human', 2, (128, 0, 128)),
        KITTI('cyclist', 6, 6, 'human', 2, (128, 128, 128)),
        KITTI('tram', 7, 7, 'vehicle', 1, (64, 192, 0)),
        KITTI('misc', 8, 8, 'void', 0, (192, 0, 0)),
    ]
    """
        KITTI('static', 9, 255, 'void', 0, (0, 0, 0)),
        KITTI('dynamic', 10, 255, 'void', 0, (128, 0, 229)),"""

    train_id_to_color = [c.color for c in classes if c.train_id != 255]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])

    def __init__(self, file, new_size, ):
        # self.split = file.split('/')[-1].split('.')[0]

        self.resize_factor = new_size  # resize之后图像和深度图的H,W
        self.paths = []
        with open(file, 'r') as f:
            data = f.readlines()
        for i, fname in enumerate(data):
            path = fname.split()[0]
            if os.path.exists(path):
                self.paths.append(path)

    ####################################################################################################################
    # @为python装饰器：可以让某个函数在不改动代码的基础上增加额外的功能。比如函数的嵌套：staticmethod(func)
    @staticmethod
    def _get_next_file(idx, file):
        # 返回的是当前文件路径下往后数idx帧的路径
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))  # base=0000000000,ext=.png
        #判断该文件路径是否存在
        return os.path.join(os.path.dirname(file), str(idx+int(base)).zfill(len(base)) + ext)  # zfill是向右对齐添零

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../.."))  # 往上数第四层文件夹的路径:这里的用法还存在疑问

    @staticmethod
    def _get_geometic_information(source, target):
        """找到图像对应的深度图和姿态所在的位置"""
        folder = os.path.abspath(os.path.join(source, "../../.."))
        base1 = os.path.splitext(os.path.basename(source))[0]
        base2 = os.path.splitext(os.path.basename(target))[0]
        depth1_file = folder+'/depth/'+base1+'.npz'
        depth2_file = folder+'/depth/'+base2+'.npz'
        pose_file = folder+'/pose/'+base2+'.npz'
        # npz里保存的数组名为'arr_0'
        return np.load(depth1_file)['arr_0'], np.load(depth2_file)['arr_0'], np.load(pose_file)['arr_0']


    @staticmethod
    def _get_intrinsics(image_file, calib_data):
        """Get intrinsics from the calib_data dictionary."""
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return intrinsics
            if IMAGE_FOLDER[cam] in image_file:
                # 返回的是cam_cam的配准文件中P_rect_02和P_rect_03两行信息即uv坐标系转到相机2、3坐标系的内参
                return np.reshape(calib_data[IMAGE_FOLDER[cam].replace('image', 'P_rect')], (3, 4))[:, :3]

    @staticmethod
    def _read_raw_calib_file(folder):
        # 这里读取的不同的相机坐标系之间的配准文件，将每一行的数据转换成矩阵，返回字典型数据
        """Read raw calibration files from folder."""
        return read_calib_file(os.path.join(folder, CALIB_FILE['cam2cam']))

    @classmethod
    #将mask转换成rgb图像，类似于加上颜色，以便结果可视化
    def decode_target(cls, target):
        #target[target == 255] = 8
        return cls.train_id_to_color[target]

    ####################################################################################################################
    #### 动态掩膜：输出[numclasses,H,W]，并且每个通道数值只有1和0
    ####################################################################################################################
    def get_mask2onehot(self, image_file):
        mask_file = image_file.replace('image_02/data', 'mask/np_array_All_classes_Output')
        mask_file = mask_file.replace('png', 'npz')

        with np.load(mask_file) as data:
            mask = data['x']  # [H,W,C] C是instant的个数，即识别出几个物体
            if mask.shape[-1] != 2484:  # 图片不存在任何物体时，shape为[h,w]不需要再处理
                mask = np.amax(mask, axis=-1)
        return mask  # h,w


    def onehot2mask(mask):
        """
        Converts a mask (K, H, W) to (H,W)
        """
        _mask = np.argmax(mask, axis=0).astype(np.uint8)
        return _mask
    ####################################################################################################################

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information
        sample = {}
        c_f = self.paths[idx]
        l_f = self._get_next_file(-1, c_f)  # 取前一帧
        ###############################################label############################################################
        # 当前帧对应的掩膜即ground truth,转换成onehot格式，shape[H, W, 8]
        label = self.get_mask2onehot(c_f)
        # groundtruth里的mask的size为[750, 2485]，转成和图片size相同的[256, 832]
        label = skimage.transform.resize(label, self.resize_factor,
                                          order=0, mode='reflect', preserve_range=True)
        label = torch.from_numpy(label).long()
        ###############################################label############################################################
        depth_s, depth_t, pose = self._get_geometic_information(l_f, c_f)

        # camera2的内参，先找到对应日期下的calib_cam2cam.txt文件然后找到相机对应的行即内参
        intrinsics = self._get_intrinsics(c_f, self._read_raw_calib_file(self._get_parent_folder(c_f)))
        img = load_image(c_f)  # 输出[w,h]
        height_scale = img.size[1] // self.resize_factor[0]
        width_scale = img.size[0] // self.resize_factor[1]
        # resize内参
        intrinsics[0, :] /= width_scale  # fu、cu都按照宽度缩放
        intrinsics[1, :] /= height_scale  # fv、cv都按照高度缩放
        # 直接使用torch.from_numpy把矩阵转换成张量时默认是float64类型，但是默认的精度是float32，会导致后面出现精度不匹配的问题，所以需要再转换成float32
        intrinsics_ = torch.from_numpy(intrinsics).float()  # 增加batch_size维度

        #源视图是上一帧，目标视图是当前帧
        source_view = transform_rgb(load_image(l_f))  #.unsqueeze(0)
        target_view = transform_rgb(load_image(c_f))  #.unsqueeze(0)  # ([1, 3, 128, 416])

        source_depth = transform_depth(np.resize(depth_s, [128, 416]))  #.unsqueeze(0)
        target_depth = transform_depth(np.resize(depth_t, [128, 416]))  #.unsqueeze(0)  # ([1, 1, 128, 416])


        sample.update({
            'label': label,
            'Ir': source_view,
            'It': target_view,
            'Dr': source_depth,
            'Dt': target_depth,
            'pose': pose.reshape(-1),  # 轴角
            #'trans': pose[:, :3],
            'intrinsics_mat': intrinsics_,
        })
        return sample


########################################################################################################################
if __name__ == "__main__":
    file = '/home/zhanl/data/code/motion_seg/data/train.txt'
    kitti_dataset = KITTIDataset(file, (128, 416))
    train_loader = DataLoader(kitti_dataset, batch_size=16, shuffle=True, num_workers=4)
    for data in train_loader:
        labels, Ir, It, Dr, Dt, pose, intrinsics_ = (data[s] for s in
                                                                 ['label', 'Ir', 'It', 'Dr', 'Dt', 'pose',
                                                                  'intrinsics_mat'])
        print()
