# Copyright 2020 Toyota Research Institute.  All rights reserved.
# 主要任务：创建各个文件的路径（以字典、列表的格式）、读取深度信息、
# 这里的整个数据的读取流程为：（以raw dataset为例）编辑一个train_txt文件包含了训练集所有数据的路径和文件名→找到这些RGB图片对应的深度图→
import glob
import torch.multiprocessing

import numpy as np
import os

from torch.utils.data import Dataset

from dataset.original_method.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from dataset.utils.image import load_image
from dataset.utils.transforms import to_tensor_transforms, transform_pose, transform_rgb, transform_depth
from dataset.data_prepocess.Matrix_Angle import R_to_angle

########################################################################################################################
#torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

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
    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1.  # 这里没有深度的像素值为什么要改成-1：归一化到[-1，1]的区间
    return np.expand_dims(depth, axis=2)


########################################################################################################################
#### DATASET
########################################################################################################################


class KITTIDataset(Dataset):
    """
    KITTI dataset class.

    Parameters
    ----------
    root_dir : str
        Path to the dataset
    file_list : str
        Split file, with paths to the images to be used     example: 2011_09_26/2011_09_26_drive_0009_sync/0000000386.png
    train : bool
        True if the dataset will be used for training
    data_transform : Function
        Transformations applied to the sample
    depth_type : str
        Which depth type to load  类别分两种：'velodyne'和'groundtruth'（带注释）
    with_pose : bool
        True if returning ground-truth pose
    back_context : int
        Number of backward frames to consider as context ??? 上下文信息指的一般是当前像素点周围像素的信息，上下文特征即为当前像素及其周边像素之间的某种联系
    forward_context : int
        Number of forward frames to consider as context ???  这里的上下文消息可以理解为相邻帧吗？？
    strides : tuple
        List of context strides
    with_dynamic : bool
        True if return a dynamic dataset
    dynamic_list: list
        List included all the index of dynamic image  （和file_list的区别是file_list是文件路径，dynamic是存储了动态图片路径的列表，和后文的self.path类似

    """

    def __init__(self, file_list, dynamic_list,
                 with_dynamic=False, with_pose=False, to_cuda=True, half=False,
                 pre_transform=None, data_transform=to_tensor_transforms,
                 depth_type='groundtruth',
                 backward_context=0, forward_context=0, strides=(1,)):
        # Assertions
        assert backward_context >= 0 and forward_context >= 0, 'Invalid contexts'

        self.backward_context = backward_context
        self.backward_context_paths = []  # 存的id
        self.forward_context = forward_context
        self.forward_context_paths = []

        self.with_context = (backward_context != 0 or forward_context != 0)
        # file_list = "/home/zhanl/data/eigen_zhou_files.txt",self.split=eigen_zhou_files
        self.split = file_list.split('/')[-1].split('.')[0]

        self.dynamic_index = dynamic_list
        self.data_transform = data_transform

        self.with_dynamic = with_dynamic  # 判断是否是动态

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None  # 即depth_type非空即为True
        self.with_pose = with_pose

        # 这里的cache的理解：保存了上一个数据的读取记录
        self._cache = {}  # 存储的是最后以个文件夹路径的文件数量（每个序列参照深度图数量）
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        with open(file_list, "r") as f:
            data = f.readlines()  # data是列表类型的数据，一行代表一个元素

        self.paths = []
        self.depth = []

        if self.with_dynamic:
            data = self.dynamic_index  # 这里不需要筛选出来没有对应深度信息的rgb图，因为dynamic_list就是从总数据集里挑选出来的已经是满足了条件的

        # Get file list from data:需要编辑一个包含了训练集中所有数据的路径和文件名的txt文件 (这里可以再加上pose文件路径或者pose每次通过计算得到)
        for i, fname in enumerate(data):
            # 可以理解为/home/zhanlei/data/kitti-raw/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/0000000000.png
            #path = os.path.join(root_dir, fname.split()[0])  # 当split括号里不设置符号时，默认一个空格就是一次断开
            path = fname.split()[0]
            #if os.path.exists(path) and not (fname in self.dynamic_index):  # 筛除掉动态列表里的帧？
            if os.path.exists(path):
                if not self.with_depth:  # 不需要深度信息
                    self.paths.append(path)
                else:  # 如果要输入深度图片的话，再添加一个深度路径
                    # Check if the depth file exists
                    depth = self._get_depth_file(path)  # depth是该rgb图片对应的深度图路径，但也有可能找不到对应的深度图
                    if depth is not None and os.path.exists(
                            depth):  # 如果找不到当前rgb对应的depth即不把该rgb加入到路径中，即解决了深度图要比rgb图少十帧的问题
                        self.paths.append(path)
                        self.depth.append(depth)
            # 得到n个rgb图片路径（这里不需要设置一个深度图路径列表，因为深度图可直接转换rgb途径读取）和n个depth路径

        # If using context, filter file list
        if self.with_context:
            # 当strides为(1,)元组时，下面的循环只能执行一次，且stride=1
            for stride in strides:
                for idx, file in enumerate(self.paths):
                    # 这里动态池的前后帧筛选方式不一样，应该是直接在list里面往前数一个而不是idx-1
                    backward_context_idxs, forward_context_idxs = \
                        self._get_sample_context(
                            file, backward_context, forward_context, stride)
                    self.forward_context_paths.append(forward_context_idxs)
                    self.backward_context_paths.append(backward_context_idxs)  # 得到了当前帧的前后帧id

        # 对所有帧进行预处理
        self.sample = {
            'filename': [],  # eigen_zhou_files_idx左边补0至十位
            'idx': [],  # 包含了日期时间段和index
            'rgb': [],
            'depth': [],
            'pose': [],
            'rgb_last': [],
            'rgb_next': [],
            'depth_last': [],
            'depth_next': [],
            'pose_last': [],
            'pose_next': []
        }
        for index in range(len(self.paths)):
            self.sample['filename'].append(self.paths[index])
            self.sample['idx'].append(extract_idx(self.paths[index]))
            image = load_image(self.paths[index])
            self.sample['rgb'].append(transform_rgb(image))

            if self.with_pose:
                pose = self._get_pose(self.paths[idx])  # 这个时候得到的pose已经是张量了
                self.sample['pose'].append(transform_pose(pose))

            if self.with_depth:
                depth = self._read_depth(self._get_depth_file(self.paths[idx]))
                self.sample['depth'].append(transform_depth(depth))  # transform过的depth

            if self.with_context:
                all_context_idxs = self.backward_context_paths[idx] + \
                                   self.forward_context_paths[idx]  # 两个id重复的话输出的是什么
                image_context_paths, depth_context_paths = \
                    self._get_context_files(self.paths[idx], all_context_idxs)
                if self.with_dynamic:
                    if idx > 0:
                        backward_context_idxs = idx - 1
                    else:  # 动态池第一帧
                        backward_context_idxs = idx + 1
                    if idx < len(self.paths) - 1:
                        forward_context_idxs = idx + 1
                    else:  # 动态池最后一帧
                        forward_context_idxs = idx - 1
                    depth_context_paths.append(self._get_depth_file(self.paths[backward_context_idxs]))
                    depth_context_paths.append(self._get_depth_file(self.paths[forward_context_idxs]))

                image_context = [load_image(f) for f in image_context_paths]
                image_last = image_context[0]
                image_next = image_context[-1]  # 改成1的时候会报错：list index out of range
                self.sample['rgb_last'].append(transform_rgb(image_last))
                self.sample['rgb_next'].append(transform_rgb(image_next))

                depth_context = [self._read_depth(f) for f in depth_context_paths]  # depth_context需要筛除掉不存在的帧
                depth_last = depth_context[0]
                # print(depth_context_paths[0])  这里读出帧的顺序又全部错乱了，应该是上面有一部是随机读取的
                depth_next = depth_context[-1]
                # print(depth_context_paths[-1])
                self.sample['depth_last'].append(transform_depth(depth_last))
                self.sample['depth_next'].append(transform_depth(depth_next))

                # Add context poses
                if self.with_pose:  # 一般默认是True
                    #first_pose = sample['pose']
                    image_context_pose = [self._get_pose(f) for f in image_context_paths]
                    pose_last = image_context_pose[0]
                    pose_next = image_context_pose[-1]
                    self.sample['pose_last'].append(transform_pose(pose_last))
                    self.sample['pose_next'].append(transform_pose(pose_next))
        if to_cuda:
            self.sample = self.sample.to(device)


    ########################################################################################################################
    # @为python装饰器：可以让某个函数在不改动代码的基础上增加额外的功能。比如函数的嵌套：staticmethod(func)
    @staticmethod
    def _get_next_file(idx, file):
        # 返回的是当前文件向后数idx个数字后的文件路径？之前的路径怎么读取
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))  # base=0000000000,ext=.png
        return os.path.join(os.path.dirname(file), str(idx).zfill(len(base)) + ext)  # zfill是向右对齐添零

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
        # 这里读取的不同的相机坐标系之间的配准文件
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

    # sample是一个字典类型的数据，包含三个键id、图像文件名、图像。
    def _get_sample_context(self, sample_name, backward_context, forward_context, stride=1):
        """
        Get a sample context

        Parameters
        ----------
        sample_name : str
            Path + Name of the sample
        backward_context : int = 1
            Size of backward context
        forward_context : int = 1
            Size of forward context
        stride : int （步长） = 1
            Stride value to consider when building the context

        Returns
        -------
        backward_context : list of int
            List containing the indexes for the backward context
        forward_context : list of int
            List containing the indexes for the forward context
        """
        base, ext = os.path.splitext(os.path.basename(sample_name))
        parent_folder = os.path.dirname(sample_name)
        """不知道为啥做了下面这些修改后，代码运行的速度特别慢
        for cam in ['left', 'right']:
            if IMAGE_FOLDER[cam] in parent_folder:
                parent_folder_depth = parent_folder.replace(
                            IMAGE_FOLDER[cam] + '/data', 'proj_depth/{}/{}'.format(
                                self.depth_type, IMAGE_FOLDER[cam]))
                parent_folder_depth = parent_folder_depth.replace(
                    'raw', 'depth')"""
        # current_idx
        c_idx = int(base)  # base = 0000000005, c_idx = 5

        # Check number of files in folder（这里也要改成depth文件夹中的数量）
        if parent_folder in self._cache:
            max_num_files = self._cache[parent_folder]
        else:
            max_num_files = len(glob.glob(os.path.join(parent_folder, '*' + ext))) - 5  # 当前目录下全部文件数量,减5的原因是深度图最后一张图的下标要比rgb最后一张图的小标小5
            self._cache[parent_folder] = max_num_files  # idx最大的深度图要比rgb少5

        # Check bounds
        if (c_idx - backward_context * stride) < 4 or (
                c_idx + forward_context * stride) > max_num_files:
            return None, None

        # Backward context：创造一个当前帧对应的前馈图像集合，这里如果设置容量为1的话是否就满足了相似度的计算
        b_idx = c_idx
        backward_context_idxs = []
        # 这里的while改成if，这样就只循环一次，context只包含了一张图像：
        if len(backward_context_idxs) < backward_context and b_idx > 5:  # 考虑depth是从第五张开始计数的情况
            # while len(backward_context_idxs) < backward_context and c_idx > 0:
        #if b_idx > 5:
            b_idx -= stride
        else:
            b_idx += stride  # 如果没有前缀的话（比如第一张图像），那就让前缀的图像id=后缀的图像id
        filename = self._get_next_file(b_idx, sample_name)
        if os.path.exists(filename):
            backward_context_idxs.append(b_idx)  # 记录了前stride帧的id

        # Forward context：创造一个当前帧对应的后馈图像集合
        f_idx = c_idx
        forward_context_idxs = []
        if len(forward_context_idxs) < forward_context and f_idx < max_num_files-1:
            # while len(forward_context_idxs) < forward_context and c_idx < max_num_files:
        #if f_idx < max_num_files:
            f_idx += stride
        else:
            f_idx -= stride  # 如果是最后一张图像，那就让后缀id=前缀id
        filename = self._get_next_file(f_idx, sample_name)
        if os.path.exists(filename):
            forward_context_idxs.append(f_idx)  # 记录了后stride帧的id

        # 返回的是前后帧列表（列表中只包含了图片对应的index比如说0000000000）
        return backward_context_idxs, forward_context_idxs

    def _get_context_files(self, sample_name, idxs):
        """
        Returns image and depth (and pose) context files
        即每一帧都得有对应的前后帧
        Parameters
        ----------
        sample_name : str
            Name of current sample
        idxs : list of idxs
            Context indexes

        Returns
        -------
        image_context_paths : list of str
            List of image names for the context
        depth_context_paths : list of str
            List of depth names for the context
        """
        image_context_paths = [self._get_next_file(i, sample_name) for i in idxs]
        if self.with_depth:
            depth_context_paths = [self._get_depth_file(f) for f in image_context_paths]
            return image_context_paths, depth_context_paths
        else:
            return image_context_paths, None
        # 返回的是前后帧（上下文）的图片路径
        # 还要考虑前后帧对应的姿态信息

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
        cam0_cam2_rec_mat = np.vstack((cam2cam['P_rect_02'].reshape(3, 4), [0, 0, 0, 1]))  # P_rect_xx本来就是3x4的矩阵
        cam0_cam2_mat = transform_from_rot_trans(cam2cam['R_02'], cam2cam['T_02'])  # cam0转到cam2坐标系下

        # imu2cam = cam_2rect_mat @ velo2cam_mat @ imu2velo_mat  # 这里得到的是相机0对应的图像坐标系下的的转换矩阵
        imu2cam = cam0_cam2_mat @ velo2cam_mat @ imu2velo_mat  # 这里不确定是否要和矫正矩阵做乘积
        # imu2cam = cam0_cam2_rec_mat @ cam_2rect_mat @ velo2cam_mat @ imu2velo_mat

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
        """Gets the pose information from an image file."""
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
        '''odo_pose = (imu2cam @ np.linalg.inv(origin_pose) @
                    pose_imu @ np.linalg.inv(imu2cam)).astype(np.float32)  # 4*4 @ (4*4)逆 @ (4*4) @ (4*4)逆=4*4 即旋转矩阵不包含位置信息'''

        # 摘自https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py和https://stackoverflow.com/questions/66672284/compute-pose-transformation-matrix-rotation-and-translation-from-gps-locatio
        pose_rot = transform_from_rot_trans(R, t - origin_t)  # 这里的pose都是针对第一帧的变化！！！！！
        pose = R_to_angle(pose_rot)
        pose = torch.from_numpy(pose)
        pose = pose.to(torch.float32)# 得到的是六自由度姿态：x、y、z方向平移距离和角度偏移值
        """odo_pose1 = (imu2cam @ np.linalg.inv(origin_pose1) @
                    #pose1 @ np.linalg.inv(imu2cam)).astype(np.float32)  # 3*4 @ (3*2)逆 @ (3*2) @ (3*4)逆"""

        # Cache and return pose
        # 这里把针对每个图片计算出来的pose也加入缓存中，不需要重复计算
        self.pose_cache[image_file] = pose
        return pose

    ########################################################################################################################

    def __len__(self):
        """Dataset length."""
        return len(self.paths)

    def __getitem__(self, idx):
        """Get dataset sample given an index."""
        # Add image information

        sample = {
            'filename': self.sample['filename'][idx],  # eigen_zhou_files_idx左边补0至十位
            'idx': self.sample['idx'][idx],  # 包含了日期时间段和index
            'rgb': self.sample['rgb'][idx],
            'depth': self.sample['depth'][idx],
            'pose': self.sample['pose'][idx],
            'rgb_last': self.sample['rgb_last'][idx],
            'rgb_next': self.sample['rgb_next'][idx],
            'depth_last': self.sample['depth_last'][idx],
            'depth_next': self.sample['depth_next'][idx],
            'pose_last': self.sample['pose_last'][idx],
            'pose_next': self.sample['pose_next'][idx]

        }


        # Apply transformations
        if self.data_transform:
            sample = self.data_transform(sample)

        return sample


########################################################################################################################
if __name__ == "__main__":
    kitti_dataset = KITTIDataset(file_list='/home/zhanl/data/kitti/data_splits/train.txt',dynamic_list=None,
                                 with_dynamic=False, with_pose=True, to_cuda=True, half=False,
                                 pre_transform=True, data_transform=None,
                                 forward_context=1,
                                 backward_context=1,
                                 depth_type='groundtruth')  # resize的时候改成等比例缩放

    print(len(kitti_dataset))
