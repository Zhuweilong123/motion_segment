# Copyright 2020 Toyota Research Institute.  All rights reserved.
# 主要任务：创建各个文件的路径（以字典、列表的格式）、读取深度信息、
# 这里的整个数据的读取流程为：（以raw dataset为例）编辑一个train_txt文件包含了训练集所有数据的路径和文件名→找到这些RGB图片对应的深度图→
import skimage
import torch.multiprocessing

import numpy as np
import os
import cv2

from torch.utils.data import Dataset

from dataset.original_method.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from dataset.utils.image import load_image
from dataset.data_prepocess.Matrix_Angle import isRotationMatrix, euler_from_matrix
from dataset.utils.warper import inverse_warp2, dp2flow
from dataset.utils.transforms import transform_rgb, transform_depth, transform_bwarp

########################################################################################################################
'''seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)'''
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("using {} device.".format(device))

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
def extract_idx(path):
    day = path.split('/')[-4].split('_')
    index = path.split('/')[-1].split('.')[0]
    #print(index)
    index = int(day[1] + day[2] + day[4] + index)
    return index


def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)


def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)
    assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.  # 这里没有深度的像素值为什么要改成-1：归一化到[-1，1]的区间
    return np.expand_dims(cv2.resize(depth, (832, 256)), axis=2)  # cv2.resize排序方式是W,H


########################################################################################################################
#### DATASET
########################################################################################################################

class KITTIDataset(Dataset):
    def __init__(self, file, depth_type='groundtruth'):

        # self.split = file.split('/')[-1].split('.')[0]
        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None  # 即depth_type非空即为True

        self.resize_factor = (256, 832)  # resize之后图像和深度图的H,W

        self.paths = []
        with open(file, 'r') as f:
            data = f.readlines()
        for i, fname in enumerate(data):
            path = fname.split()[0]
            if os.path.exists(path):
                """if not self.with_depth:  # 不需要深度信息
                    self.list.append(path)
                else:"""
                # Check if the depth file exists
                depth = self._get_depth_file(path)
                if depth is not None and os.path.exists(depth):
                    self.paths.append(path)

        # 这里的cache的理解：保存了上一个数据的读取记录
        self._cache = {}  # 存储的是最后以个文件夹路径的文件数量（每个序列参照深度图数量）
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

    ####################################################################################################################
    # @为python装饰器：可以让某个函数在不改动代码的基础上增加额外的功能。比如函数的嵌套：staticmethod(func)
    @staticmethod
    def _get_next_file(idx, file):
        # 返回的是当前文件路径下往后数idx帧的路径
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))  # base=0000000000,ext=.png
        return os.path.join(os.path.dirname(file), str(idx+int(base)).zfill(len(base)) + ext)  # zfill是向右对齐添零

    @staticmethod
    def _get_parent_folder(image_file):
        """Get the parent folder from image_file."""
        return os.path.abspath(os.path.join(image_file, "../../../../.."))  # 往上数第四层文件夹的路径:这里的用法还存在疑问

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

    ########################################################################################################################
    #### DEPTH
    ########################################################################################################################

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if self.depth_type in ['velodyne']:
            return read_npz_depth(depth_file, self.depth_type)
        elif self.depth_type in ['groundtruth']:
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    def _get_depth_file(self, image_file):
        """Get the corresponding depth file from an image file."""
        # 这里有个问题是depth和image图片总数不是一一对应的（depth一般从5开始而且每个时间序列的结尾也比rgb少五张）
        for cam in ['left', 'right']:
            if IMAGE_FOLDER[cam] in image_file:
                depth_file = image_file.replace(
                    IMAGE_FOLDER[cam] + '/data', 'proj_depth/{}/{}'.format(
                        self.depth_type, IMAGE_FOLDER[cam]))
                depth_file = depth_file.replace(
                    'raw', 'depth')
                if self.depth_type not in PNG_DEPTH_DATASETS:
                    depth_file = depth_file.replace('png', 'npz')
                return depth_file

    ########################################################################################################################
    #### 动态掩膜(更换文件路径)
    ########################################################################################################################
    def get_mask2onehot(self, image_file):
        mask_file = image_file.replace('image_02/data','mask/np_array_moving_Output')
        mask_file = mask_file.replace('png', 'npz')
        with np.load(mask_file) as data:
            mask = data['x']  # h*w*number
            mask = np.sum(mask, axis=-1)  # 把instance通道合成
   # def mask2onehot(mask, num_classes):
        """
        Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
        hot encoding vector
        """
        num_classes = 8  # 一共八个类别
        _mask = [mask == i for i in range(num_classes)]  # 得到num_classes*H*W大小只包含了truefalse的数组
        tran_mask = np.transpose(_mask, (1, 2, 0))  # N*H*W → H*W*N
        return np.array(tran_mask).astype(np.uint8)  # 将true和false转换成1和0

    def onehot2mask(mask):
        """
        Converts a mask (K, H, W) to (H,W)
        """
        _mask = np.argmax(mask, axis=0).astype(np.uint8)
        return _mask

    ########################################################################################################################
    #### 判断当前帧的前后相邻帧是否存在，这里因为有些边界帧没有深度图，结合一下就是当前帧对应的深度图的相邻帧是否存在
    ########################################################################################################################
    def judge_adjacent(self, last, next):
        if os.path.exists(self._get_depth_file(next)) and not os.path.exists(self._get_depth_file(last)):
            # next存在last不存在
            last = next
        elif os.path.exists(self._get_depth_file(last)) and not os.path.exists(self._get_depth_file(next)):
            next = last
        return last, next


    ########################################################################################################################
    #### POSE:要考虑的是以哪个点为原点
    ########################################################################################################################

    def _get_imu2cam_transform(self, image_file):
        """Gets the transformation between IMU and camera from an image file"""
        parent_folder = self._get_parent_folder(image_file)
        if image_file in self.imu2velo_calib_cache:
            return self.imu2velo_calib_cache[image_file]

        cam2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['cam2cam']))
        imu2velo = read_calib_file(os.path.join(parent_folder, CALIB_FILE['imu2velo']))
        velo2cam = read_calib_file(os.path.join(parent_folder, CALIB_FILE['velo2cam']))

        velo2cam_mat = transform_from_rot_trans(velo2cam['R'], velo2cam['T'])
        imu2velo_mat = transform_from_rot_trans(imu2velo['R'], imu2velo['T'])
        # 摄像机00矫正：3x3 纠正旋转矩阵(使图像平面共面)，可理解转换到2D坐标系中必须进行的
        cam_2rect_mat = transform_from_rot_trans(cam2cam['R_rect_00'], np.zeros(3))
        # 00到02坐标系
        cam0_cam2_rec_mat = np.vstack((cam2cam['P_rect_02'].reshape(3, 4), [0, 0, 0, 1]))  # P_rect_xx本来就是3x4的矩阵
        #cam0_cam2_mat = transform_from_rot_trans(cam2cam['R_02'], cam2cam['T_02'])  # cam0转到cam2坐标系下

        # imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat  # 这里得到的是相机0对应的图像坐标系下的的转换矩阵
        #cam2imu = imu2velo_mat @ velo2cam_mat @ cam_2rect_mat @ cam0_cam2_mat  # 这里不确定是否要和矫正矩阵做乘积
        #为什么这里是反着计算的：因为每个坐标系转换时都是左乘的
        imu2cam = cam0_cam2_rec_mat @ cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

        # 即缓存好当前日期的内参值（imu to cam2）就不用一直计算了
        self.imu2velo_calib_cache[image_file] = imu2cam
        return imu2cam

    @staticmethod
    def _get_oxts_file(image_file):
        """Gets the oxts file from an image file."""
        # find oxts pose file
        for cam in ['left', 'right']:
            # Check for both cameras, if found replace and return file name
            if IMAGE_FOLDER[cam] in image_file:
                return image_file.replace(IMAGE_FOLDER[cam], OXTS_POSE_DATA).replace('.png', '.txt')
        # Something went wrong (invalid image file)
        raise ValueError('Invalid KITTI path for pose supervision.')

    def _get_oxts_data(self, image_file):
        """Gets the oxts data from an image file."""
        oxts_file = self._get_oxts_file(image_file)
        if oxts_file in self.oxts_cache:
            oxts_data = self.oxts_cache[oxts_file]
        else:
            oxts_data = np.loadtxt(oxts_file, delimiter=' ', skiprows=0)
            self.oxts_cache[oxts_file] = oxts_data
        return oxts_data

    def _get_pose(self, image_file):
        """Gets the pose information from an image file.
            基于第一帧的改成帧间的pose
        """
        if image_file in self.pose_cache:
            return self.pose_cache[image_file]

        # Find origin frame in this sequence to determine scale & origin translation
        base, ext = os.path.splitext(os.path.basename(image_file))
        origin_frame = os.path.join(os.path.dirname(image_file), str(0).zfill(len(base)) + ext)  # 每个时间序列的0000000000.png

        # Get origin data
        origin_oxts_data = self._get_oxts_data(origin_frame)
        lat = origin_oxts_data[0]
        scale = np.cos(lat * np.pi / 180.)
        # Get origin pose
        origin_R, origin_t = pose_from_oxts_packet(origin_oxts_data, scale)

        origin_pose = transform_from_rot_trans(origin_R, origin_t)
        # origin_pose1 = transform_from_angle_trans(origin_t, origin_angle)
        # 这里的用处不大在于转换矩阵全是3X3的，无法和1X6的姿态做乘积，所以还是要在得到转换坐标系后的旋转矩阵后调用算法重新计算偏移角度（不知道会增加多大的计算复杂度）
        # Compute current pose
        oxts_data = self._get_oxts_data(image_file)
        R, t = pose_from_oxts_packet(oxts_data, scale)

        pose_imu = transform_from_rot_trans(R, t)  # 4x4的转换矩阵，如果需要欧拉角的数据形式的话要对上一步进行修改
        # pose1 = transform_from_angle_trans(t, angle)
        # Compute odometry pose
        imu2cam = self._get_imu2cam_transform(image_file)
        # @是矩阵乘法运算符
        #为什么顺序是反过来的：理解为cam0_cam2_rec_mat @ cam_2rect_mat @ velo2cam_mat @ imu2velo_mat @ imu—pose(从右往左依次转换，先转换成velo再转换成cam0最后转换成cam2)
        #还有最后一个问题：不知道这里为什么要乘以imu2cam的逆
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose_imu @ np.linalg.inv(imu2cam)).astype(np.float32)  # 4*4 @ (4*4)逆 @ (4*4) @ (4*4)逆=4*4 即旋转矩阵不包含位置信息'''
        odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose_imu).astype(np.float32)
        # 摘自https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py和https://stackoverflow.com/questions/66672284/compute-pose-transformation-matrix-rotation-and-translation-from-gps-locatio
        """pose_rot = transform_from_rot_trans(R, t - origin_t)  # 这里的pose都是针对第一帧的变化！！！！！
        pose = R_to_angle(pose_rot)
        pose = torch.from_numpy(pose)
        pose = pose.to(torch.float32)# 得到的是六自由度姿态：x、y、z方向平移距离和角度偏移值"""


        # Cache and return pose
        # 这里把针对每个图片计算出来的pose也加入缓存中，不需要重复计算
        self.pose_cache[image_file] = odo_pose[:-1, :]  # 把最后一行(齐次坐标系)删掉
        return odo_pose[:-1, :]

    # 全局pose改成帧间pose（这里的pose都是3*4的矩阵形式）
    # 求出1→2的转换矩阵
    def interframe_pose(self, mat1, mat2):
        # frame 2
        # rotation matrix
        matrix_R_n = mat2[:, :3]
        # translation vector
        matrix_t_n = mat2[:, 3:]

        # frame 1
        matrix_R_n_ = mat1[:, :3]
        matrix_t_n_ = mat1[:, 3:]

        # Inverse matrix of frame 1
        inverse_matrix_R_n_ = np.linalg.inv(matrix_R_n_)

        # compute the pose between the frames
        t = inverse_matrix_R_n_ @ (matrix_t_n - matrix_t_n_)
        # t = matrix_t_n - matrix_t_n_
        R = inverse_matrix_R_n_ @ matrix_R_n

        assert (isRotationMatrix(R))
        # 返回3*4的shape
        return transform_from_rot_trans(R, t)


    ########################################################################################################################

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information
        sample = {}
        c_f = self.paths[idx]  # 当前路径

        depth = self._read_depth(self._get_depth_file(c_f))
        # 这里得到的pose就是转换矩阵的格式[3,4]
        pose = self._get_pose(c_f)

        l_f = self._get_next_file(-1, c_f)
        n_f = self._get_next_file(1, c_f)

        # 判断是否前一帧和后一帧都存在
        l_f, n_f = self.judge_adjacent(l_f, n_f)
        depth_l = self._read_depth(self._get_depth_file(l_f))
        pose_l = self._get_pose(l_f)

        depth_n = self._read_depth(self._get_depth_file(n_f))
        pose_n = self._get_pose(n_f)

        # camera2的内参，先找到对应日期下的calib_cam2cam.txt文件然后找到相机对应的行即内参
        intrinsics = self._get_intrinsics(c_f, self._read_raw_calib_file(self._get_parent_folder(c_f)))
        height_scale = depth.shape[0] // self.resize_factor[0]
        width_scale = depth.shape[1] // self.resize_factor[1]
        # resize内参
        intrinsics[0, :] /= width_scale
        intrinsics[1, :] /= height_scale
        # 直接使用torch.from_numpy把矩阵转换成张量时默认是和np的精度相同，np是float64转换得到的tenor也是float64的，但是torch默认的精度是float32，会导致后面出现精度不匹配的问题，所以需要再转换成float32
        intrinsics_ = torch.from_numpy(intrinsics).unsqueeze(0).float()  # 增加batch_size维度

        # 源视图是上一帧，目标视图是当前帧
        source_view = transform_rgb(load_image(l_f)).unsqueeze(0)
        target_view = transform_rgb(load_image(c_f)).unsqueeze(0)

        source_depth = transform_depth(depth_l).unsqueeze(0)
        target_depth = transform_depth(depth).unsqueeze(0)

        # 得到前一帧到当前帧的转换矩阵
        s_t_pose = self.interframe_pose(pose_l, pose)
        # hstack拼接的两个矩阵维度ndim应该相同(2维)，reshape之前是(3,)即一维
        # 转换矩阵[R|t]的逆为[R'|-R'@t]
        t_s_pose = np.hstack([s_t_pose[:, :-1].T, -(s_t_pose[:, :-1].T @ s_t_pose[:, -1]).reshape(3, 1)])

        # 3x3: source to target的pose
        rot_ = torch.from_numpy(euler_from_matrix(s_t_pose[:, :-1]))
        # 3x1
        trans_ = torch.from_numpy(s_t_pose[:, -1])

        s_t_pose = torch.from_numpy(s_t_pose).unsqueeze(0)
        t_s_pose = torch.from_numpy(t_s_pose).unsqueeze(0)

        # 得到的是source warp 到target plane的合成图像与真实图像的像素差(这所有输入需要转换成张量)
        warped_s_t = abs(inverse_warp2(source_view, target_depth, source_depth, t_s_pose, intrinsics=intrinsics_,
                                       rotation_mode='translation', padding_mode='zeros')[0] - target_view)
        # 得到的是target warp到source plane的合成图像
        warped_t_s = abs(inverse_warp2(target_view, source_depth, target_depth, s_t_pose, intrinsics=intrinsics_,
                                       rotation_mode='translation', padding_mode='zeros')[0] - source_view)
        # 由姿态信息和深度信息合成的flow
        #flow_s_t = dp2flow(target_depth, t_s_pose, intrinsics, rotation_mode='translation')
        #flow_t_s = dp2flow(source_depth, s_t_pose, intrinsics, rotation_mode='translation')

        ###############################################label############################################################
        # 当前帧对应的掩膜即ground truth,转换成onehot格式，shape[H, W, 8]
        label = self.get_mask2onehot(c_f)
        # groundtruth里的mask的size为[750, 2485]，转成和图片size相同的[256, 832]
        label1 = skimage.transform.resize(label, self.resize_factor,
                                          order=0, mode='reflect', preserve_range=True)
        label1 = torch.from_numpy(label1).float()

        # Generate different label scales
        label2 = skimage.transform.resize(label, (label.shape[0] // 4, label.shape[1] // 4),
                                          order=0, mode='reflect', preserve_range=True)
        label2 = torch.from_numpy(label2).float()

        label3 = skimage.transform.resize(label, (label.shape[0] // 8, label.shape[1] // 8),
                                          order=0, mode='reflect', preserve_range=True)
        label3 = torch.from_numpy(label3).float()

        label4 = skimage.transform.resize(label, (label.shape[0] // 16, label.shape[1] //16),
                                          order=0, mode='reflect', preserve_range=True)
        label4 = torch.from_numpy(label4).float()

        label5 = skimage.transform.resize(label, (label.shape[0] // 32, label.shape[1] //32),
                                          order=0, mode='reflect', preserve_range=True)
        label5 = torch.from_numpy(label5).float()
        ###############################################label############################################################

        rgb_ = torch.cat((target_view, source_view), dim=0)

        depth_ = torch.cat((target_depth, source_depth), dim=0)

        # 网络的输入
        #static_flow = torch.cat((transform_fwarp(flow_s_t), transform_fwarp(flow_t_s)), dim=0)
        warp_image = torch.cat((warped_s_t, warped_t_s), dim=0)
        # 可视化视差图像


        sample.update({
            'rgb': rgb_,
            'rot': rot_,
            'trans': trans_,
            'intrinsics': intrinsics_,
            'depth': depth_,
            #'flo': static_flow,
            'warped': warp_image,
            'label1': label1,
            'label2': label2,
            'label3': label3,
            'label4': label4,
            'label5': label5,
        })

        return sample





"""
将数据提前载入到cuda上
class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()

        self.preload()

    def preload(self):
        try:
            self.p1, self.d1, self.p2, self.d2, self.pp, self.dp, self.pn, self.dn = next(self.loader)
        except StopIteration:
            self.p1 = None
            self.d1 = None
            self.p2 = None
            self.d2 = None
            self.pp = None
            self.dp = None
            self.pn = None
            self.dn = None
            return
        with torch.cuda.stream(self.stream):
            #self.sample = self.sample.cuda(non_blocking=True)
            self.p1 = self.p1.to(device, non_blocking=True)
            self.d1 = self.d1.to(device, non_blocking=True)
            self.p2 = self.p2.to(device, non_blocking=True)
            self.d2 = self.d2.to(device, non_blocking=True)
            self.pp = self.pp.to(device, non_blocking=True)
            self.dp = self.dp.to(device, non_blocking=True)
            self.dn = self.dn.to(device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)

        self.preload()
        return self.p1, self.d1, self.p2, self.d2, self.pp, self.dp, self.pn, self.dn
"""

########################################################################################################################
if __name__ == "__main__":
    file = '/home/zhanl/data/kitti/data_split_seg/train_val.txt'
    kitti_dataset = KITTIDataset(file,
                                 depth_type='groundtruth')

    print(len(kitti_dataset))
