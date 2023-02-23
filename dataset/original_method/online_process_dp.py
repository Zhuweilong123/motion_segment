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

from dataset.utils.warper import inverse_warp2
from dataset.utils.transforms import transform_rgb, transform_depth
from monodepth2 import networks

########################################################################################################################

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("using {} device.".format(device))
# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4
# Cameras from the stero pair (left is the origin)
IMAGE_FOLDER = {
    'left': 'image_02',
    'right': 'image_03',
}
'''
IMAGE_FOLDER = {
    'left': 'image_02',
}'''
# Name of different calibration files
CALIB_FILE = {
    'cam2cam': 'calib_cam_to_cam.txt',
    'velo2cam': 'calib_velo_to_cam.txt',
    'imu2velo': 'calib_imu_to_velo.txt',
}
PNG_DEPTH_DATASETS = ['groundtruth']
OXTS_POSE_DATA = 'oxts'


########################################################################################################################
#### FUNCTIONS
########################################################################################################################
def disp_to_depth(disp, min_depth, max_depth):  # 0.1, 100
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth  # 0.01
    max_disp = 1 / min_depth  # 10
    scaled_disp = min_disp + (max_disp - min_disp) * disp  # 0.01+9.99*disp
    depth = 1 / scaled_disp
    return scaled_disp, depth
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
                #channel_index = np.where(np.amax(mask, axis=(0, 1)) == 8)
                #mask[:, :, channel_index] = 0  # 将类别number为8的全部改成0
                #mask = np.sum(mask, axis=-1)
                mask = np.amax(mask, axis=-1)
        return mask  # h,w
    """
    # 实例级别→像素级别
    height, width = mask.shape[0], mask.shape[1]
     if mask.shape[-1] == 2484:  # 图片不存在所给类别的标签时，shape为[h,w]
         num_instances = 0
     else:
         num_instances = mask.shape[-1]
     new_label = np.zeros((height, width, self.num_classes), dtype=np.float32)

     for j in range(num_instances):
         # 每个instance通道的标签
         instance_label = mask[:, :, j]
         # 逐一计算每个通道
         class_indices = np.unique(instance_label)  # 记录了0和number
         class_indices = int(class_indices[class_indices != 0][0])  # 0表示背景需要删除，并读取其中唯一的元素,TODO 本来就只有0的怎么办
         new_label[:, :, class_indices] += instance_label  # [h,w,num_classes]
     """

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
        #######################################load model for monodepth2################################################
        """Function to predict for a single image or folder of images and predict pose."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model_path = os.path.join("/home/zhanl/data/code/motion_seg/monodepth2/models", "mono_640x192")
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # LOADING PRETRAINED MODEL of depth
        encoder = networks.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=device)

        # extract the height and width of image that this model was trained with
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
        encoder.load_state_dict(filtered_dict_enc)
        encoder.to(device)
        encoder.eval()

        depth_decoder = networks.DepthDecoder(
            num_ch_enc=encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)

        depth_decoder.to(device)
        depth_decoder.eval()

        # loarding model of pose
        pose_encoder_path = os.path.join(model_path, "pose_encoder.pth")
        pose_decoder_path = os.path.join(model_path, "pose.pth")

        pose_encoder = networks.ResnetEncoder(18, False, 2)
        pose_encoder.load_state_dict(torch.load(pose_encoder_path))

        pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
        pose_decoder.load_state_dict(torch.load(pose_decoder_path))

        pose_encoder.to(device)
        pose_encoder.eval()
        pose_decoder.to(device)
        pose_decoder.eval()
        ################################################################################################################
        ################################################################################################################

        # PREDICTING ON EACH IMAGE IN TURN
        with torch.no_grad():
            # Load image and preprocess
            input_image_s = pil.open(l_f).convert('RGB')
            original_width, original_height = input_image_s.size  # [1242, 375]
            input_image_s = input_image_s.resize((feed_width, feed_height), pil.LANCZOS)
            input_image_s = transforms.ToTensor()(input_image_s).unsqueeze(0)

            input_image_t = pil.open(c_f).convert('RGB')
            input_image_t = input_image_t.resize((feed_width, feed_height), pil.LANCZOS)
            input_image_t = transforms.ToTensor()(input_image_t).unsqueeze(0)

            # PREDICTION depth
            input_image_s = input_image_s.to(device)
            input_image_t = input_image_t.to(device)
            outputs_s = depth_decoder(encoder(input_image_s))
            outputs_t = depth_decoder(encoder(input_image_t))

            disp1 = outputs_s[("disp", 0)]
            disp2 = outputs_t[("disp", 0)]

            scaled_disp_s, depth_s = disp_to_depth(disp1, 0.1, 100)  # 这里的深度应该是相对深度，因为没有相机到真实场景的映射关系
            scaled_disp_t, depth_t = disp_to_depth(disp2, 0.1, 100)
            #print(depth_s.shape) [1, 1, 192, 640]
            depth_s = STEREO_SCALE_FACTOR * depth_s.cpu().numpy()
            depth_t = STEREO_SCALE_FACTOR * depth_t.cpu().numpy()
            """if args.pred_metric_depth:
                metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()"""
            # prediction pose
            image = torch.cat((input_image_s, input_image_t), 1)
            axisangle, translation = pose_decoder([pose_encoder(image)])


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


        # 得到的是source图像投影到target图像坐标系下：Is→t:
        # 这个过程可以放到gpu上（dataloader提取后）去做
        #warped_s_t = inverse_warp2(source_view, target_depth, source_depth, pose, intrinsics=intrinsics_,
                                      #rotation_mode='translation', padding_mode='zeros')[0]
        # 得到的是source deoth投影到target坐标系下：Ds→t
        #warped_depth = inverse_warp2(source_view, target_depth, source_depth, pose, intrinsics=intrinsics_,
                                      #rotation_mode='translation', padding_mode='zeros')[2]
        #image = torch.cat((target_view.squeeze(0), warped_s_t.squeeze(0)), dim=0)  # 再把batch维度去除

        # 将一维深度图转换成三维（通过重复三次），然后将batch维度去除，两张深度图拼接起来
        #target_depth = torch.repeat_interleave(target_depth, 3, dim=1).squeeze(0)
        #warped_depth = torch.repeat_interleave(warped_depth, 3, dim=1).squeeze(0)
        #depth = torch.cat((target_depth, warped_depth), dim=0)

        sample.update({
            #'rgb': image,
            #'depth': depth,
            'label': label,
            # 为了重新投影并计算motion loss函数,
            'Ir': source_view,
            'It': target_view,
            'Dr': source_depth,
            'Dt': target_depth,
            'axisangle': axisangle[:, 0].squeeze(0),  # 轴角
            'trans': translation[:, 0].squeeze(0),
            'intrinsics_mat': intrinsics_,
            'path': c_f,

        })
        return sample


########################################################################################################################
if __name__ == "__main__":
    file = '/home/zhanl/data/code/motion_seg/data/train.txt'
    kitti_dataset = KITTIDataset(file, (128, 416))
    train_loader = DataLoader(kitti_dataset, batch_size=16, shuffle=True, num_workers=4)
    for data in train_loader:
        labels, Ir, It, Dr, Dt, axisangle, trans, intrinsics_ = (data[s] for s in
                                                                 ['label', 'Ir', 'It', 'Dr', 'Dt', 'axisangle', 'trans',
                                                                  'intrinsics_mat'])
