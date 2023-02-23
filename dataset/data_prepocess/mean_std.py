# 利用pytorch计算图像数据集的均值和方差
import torch
import glob
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import os

from dataset.original_method.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from dataset.data_prepocess.Matrix_Angle import isRotationMatrix, euler_from_matrix
from dataset.utils.image import load_image
from dataset.utils.misc import filter_dict
from dataset.utils.warper import inverse_warp
from dataset.utils.transforms import resize_depth

torch.multiprocessing.set_sharing_strategy('file_system')
########################################################################################################################
#### FUNCTIONS
########################################################################################################################

'''
# 找到动态集中最近的帧
def find_nearest_idx(number, dlist):  # mylist为128x1的张量（这里张量也可以和列表数据进行计算），dlist就是动态数据池
    original = []
    list_sorted = []
    for j in dlist:
        original.append(abs(number-j))
        list_sorted.append(abs(number-j))
    list_sorted.sort()
    #排除动态池中已有该帧的情况
    if list_sorted[0] != 0:
        nearest = original.index(list_sorted[0])  # 当数值一样的时候会返回对应的第一个下标
        nearest_2 = original.index(list_sorted[1])
    else:
        nearest = original.index(list_sorted[1])
        nearest_2 = original.index(list_sorted[2])
    return nearest, nearest_2  # 返回的是两个列表，存储的是动态池中索引下标，nearest是128x1的列表，表示128个数据对应的正样本，nearest_2是128x1的列表，表示128个数据对应的负样本


def find_path(idx):
    roo_dir = '/home/zhanl/data/kitti/raw//'
    seq = str(idx)
    if idx // 10 ** 16 > 9:
        i = 1
        seq1 = seq[0:2]  # 10
    else:
        i = 0
        seq1 = '0' + seq[0]  # 09
    seq2 = seq[i + 1:i + 3]  # 日期
    seq3 = seq[i + 3:i + 7]
    seq4 = seq[i + 7:i + 17]
    path = roo_dir + '2011_' + seq1 + '_' + seq2 + '/' + '2011_' + seq1 + '_' + seq2 + '_drive_' + seq3 + '_sync/image_02/data/' + seq4 + '.png'
    return path
'''


def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    transform1 = transforms.Compose([  # RGB图像预处理
        transforms.Resize([256, 832]),
        #transforms.Resize([128, 416]),
        transforms.ToTensor(),  # [C,H,W]
        #transforms.Pad([78, 0]),  # 上下填充78
        # transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.377, 0.402, 0.383], std=[0.304, 0.315, 0.319]),
    ])

    transform2 = transforms.Compose([  # 深度图像预处理
        # transforms.ToPILImage(),  # 这里又会自动转变为RGB图像
        # transforms.Grayscale(1),  转换成灰度图
        #resize_depth(),
        transforms.ToTensor(),
        #transforms.Pad([78, 0]),
        # transforms.CenterCrop(224),
        # 将其先由HWC转置为CHW格式，再转为float后每个像素除以255：和kitti深度图像预处理操作一样，这里有一个问题是只对uint8才除以255，所以这里的操作仅为由HWC转置为CHW格式
        # transfer16_01,
        transforms.Normalize((1.734,), (8.001,)),
    ])

    # Convert single items
    for key in filter_dict(sample, ['rgb']):
        sample[key] = transform1(sample[key]).type(tensor_type)
    for key in filter_dict(sample, ['depth']):
        sample[key] = transform2(sample[key]).type(tensor_type)

    # Return converted sample
    return sample


"""
def to_tensor_transforms(sample):
    sample = to_tensor_sample(sample)
    return sample
"""


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
    depth = depth_png.astype(np.float64) / 256.
    depth[depth_png == 0] = -1.  # 这里没有深度的像素值为什么要改成-1：归一化到[-1，1]的区间
    return np.expand_dims(depth, axis=2)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def transfer16_8(img):
    img_min = np.min(img)
    img_max = np.max(img)
    img_8bit = np.array(np.rint(255 * ((img - img_min) / (img_max - img_min))), dtype=np.uint8)
    return img_8bit


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



class KITTIDataset(Dataset):
    # with_dynamic代表是否有对应的动态帧，dynamic_list包含了训练集中所有动态帧的路径
    def __init__(self, file, data_transform, depth_type='groundtruth'):
                 #backward_context=0, forward_context=0, strides=(1,)
        # Assertions

        #self.split = file.split('/')[-1].split('.')[0]
        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None  # 即depth_type非空即为True

        self.data_transform = data_transform

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
        """
        self.dynamic_list = []
        self.dynamic_index = []  # 记录了索引值

        
        with open(dynamic_file, 'r') as f:
            data = f.readlines()
        for i, fname in enumerate(data):
            dynamic = fname.split()[0]
            if os.path.exists(dynamic):
                if not self.with_depth:  # 不需要深度信息
                    self.dynamic_list.append(dynamic)
                else:
                    # Check if the depth file exists
                    depth = self._get_depth_file(dynamic)
                    if depth is not None and os.path.exists(depth):
                        self.dynamic_list.append(dynamic)
        for i in range(len(self.dynamic_list)):
            self.dynamic_index.append(extract_idx(self.dynamic_list[i]))  #dynamic_index记录的都是路径对应的整数"""

        '''self.static_list = []
        self.static_index = []

        with open(static_file, 'r') as f:
            data2 = f.readlines()
        for i, fname in enumerate(data2):
            static = fname.split()[0]
            if os.path.exists(static):
                if not self.with_depth:  # 不需要深度信息
                    self.static_list.append(static)
                else:
                    # Check if the depth file exists
                    depth = self._get_depth_file(static)
                    if depth is not None and os.path.exists(depth):
                        self.static_list.append(static)
        for i in range(len(self.static_list)):
            self.static_index.append(extract_idx(self.static_list[i]))'''


        # 这里的cache的理解：保存了上一个数据的读取记录
        self._cache = {}  # 存储的是最后以个文件夹路径的文件数量（每个序列参照深度图数量）
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        #self.sequence_origin_cache = {}

        """
        with open(static_file, "r") as f:
            data = f.readlines()  # data是列表类型的数据，一行代表一个元素

        self.paths = []
        #self.labels = []  # 和self.path一一对应，记录了对应路径帧的动静态类别

        # 最近的动态帧的相似度也要小于最远的静态帧（or远远小于最近的静态帧）？？设置为hardest sample

        # Get file list from data:需要编辑一个包含了训练集中所有数据的路径和文件名的txt文件 (这里可以再加上pose文件路径或者pose每次通过计算得到)
        for i, fname in enumerate(data):
            # 可以理解为/home/zhanlei/data/kitti-raw/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/0000000000.png
            #path = os.path.join(root_dir, fname.split()[0])  # 当split括号里不设置符号时，默认一个空格就是一次断开
            path = fname.split()[0]
            #label = fname.split()[-1]
            #if os.path.exists(path) and not (fname in self.dynamic_index):  # 筛除掉动态列表里的帧？
            if os.path.exists(path):
                if not self.with_depth:  # 不需要深度信息
                    self.paths.append(path)
                    #self.labels.append(label)
                else:  # 如果要输入深度图片的话，再添加一个深度路径
                    # Check if the depth file exists
                    depth = self._get_depth_file(path)  # depth是该rgb图片对应的深度图路径，但也有可能找不到对应的深度图
                    if depth is not None and os.path.exists(
                            depth):  # 如果找不到当前rgb对应的depth即不把该rgb加入到路径中，即解决了深度图要比rgb图少十帧的问题
                        self.paths.append(path)"""
                        #self.labels.append(label)


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
        # camera2的内参，先找到对应日期下的calib_cam2cam.txt文件然后找到相机对应的行即内参
        intrinsics = self._get_intrinsics(c_f, self._read_raw_calib_file(self._get_parent_folder(c_f)))
        #label = 0  #默认当前取得anchor都是静态帧
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


        #源视图是上一帧，目标视图是当前帧
        source_view = load_image(l_f)

        target_view = load_image(c_f)

        source_depth = depth_l
        target_depth = depth
        #得到前一帧到当前帧的转换矩阵
        s_t_pose = self.interframe_pose(pose_l, pose)
        # hstack拼接的两个矩阵维度ndim应该相同(2维)，reshape之前是(3,)即一维
        # 转换矩阵[R|t]的逆为[R'|-R'@t]
        t_s_pose = np.hstack([s_t_pose[:, :-1].T, -(s_t_pose[:, :-1].T @ s_t_pose[:, -1]).reshape(3, 1)])
        """img要转成tensor即[3,H,W]
        #得到的是source warp 到target plane的合成图像与真实图像的像素差
        warped_s_t = abs(inverse_warp(source_view, target_depth, t_s_pose, intrinsics=intrinsics,
                                      rotation_mode='translation', padding_mode='zeros') - target_view)
        #得到的是target warp到source plane的合成图像
        warped_t_s = abs(inverse_warp(target_view, source_depth, s_t_pose, intrinsics=intrinsics,
                                      rotation_mode='translation', padding_mode='zeros') - source_view)"""

        # 由姿态信息和深度信息合成的flow
        #flow_s_t = dp2flow(target_depth, t_s_pose, intrinsics, rotation_mode='translation')
        #flow_t_s = dp2flow(source_depth, s_t_pose, intrinsics, rotation_mode='translation')

        """             label           """
        # 当前帧对应的掩膜即ground truth,转换成onehot格式，shape[H, W, 8]
        """label = self.get_mask2onehot(c_f)
        # groundtruth里的mask的size是比kitti数据大一倍
        label1 = skimage.transform.resize(label, (label.shape[0] // 2, label.shape[1] // 2),
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
        label5 = torch.from_numpy(label5).float()"""
        """             label           """

        #rgb_ = torch.cat((transform_rgb(target_view), transform_rgb(source_view)), dim=0)

        #depth_ = torch.cat((transform_depth(target_depth), transform_depth(source_depth)), dim=0)

        # 相机内参也要跟着resize
        #intrinsics_ = torch.from_numpy(intrinsics)
        # 3x3: source to target的pose → 欧拉角（占用小）
        #rot_ = torch.from_numpy(euler_from_matrix(s_t_pose[:, :-1]))
        # 3x1
        #trans_ = torch.from_numpy(s_t_pose[:, -1])
        # 3x4
        # transform_ = torch.from_numpy()


        # 网络的输入
        #static_flow = torch.cat((transform_fwarp(flow_s_t), transform_fwarp(flow_t_s)), dim=0)
        #warp_image = torch.cat((transform_bwarp(warped_s_t), transform_bwarp(warped_t_s)), dim=0)


        sample.update({
            # concat两个帧
            'rgb': target_view,

            'depth': resize_depth(target_depth, (256, 832)),
            #'flo': static_flow,

        })
        if self.data_transform:
            sample = self.data_transform(sample)

        '''sample.update(({
            'warped': warp_image,
        }))'''

        return sample
#####################################################################################################################################################################################################################


# 计算rgb图的均值和方差
def get_rgb_mean(train_data):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    print('Compute mean and variance for training data.')
    print(len(train_data))
    eps = 1e-5
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for _, data in enumerate(train_loader):  # _是序号，data是图像内容,读取出来的图像还要再加上一维即batchsize，最终是4维
        mean += torch.mean(data['rgb'], dim=[0, 2, 3])  # mean为[3,1]
        std += torch.std(data['rgb'], dim=[0, 2, 3])
    step = len(train_data) / 128  #循环次数，这里可能会有一点误差
    mean = mean / step
    std = std / step
    #mean = torch.mean(mean, dim=0)
    #std = torch.std(std, dim=0)
    return list(mean.numpy()), list(std.numpy())


#计算深度图的均值和方差
def get_depth_mean(train_data):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    print('Compute mean and variance for training data.')
    print(len(train_data))
    eps = 1e-5
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True)
    mean = torch.zeros(1)
    std = torch.zeros(1)
    for _, data in enumerate(train_loader):  # _是序号，data是图像内容,读取出来的图像还要再加上一维即batchsize，最终是4维
        mean += torch.mean(data['depth'], dim=[0, 2, 3])  # mean为[1,]
        std += torch.std(data['depth'], dim=[0, 2, 3])
    step = len(train_data) / 128
    mean = mean / step
    std = std / step
    return list(mean.numpy()), list(std.numpy())


# 计算视差图片的均值和方差
def get_warp_mean(train_data):
    """
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    """
    print('Compute mean and variance for training data.')
    print(len(train_data))
    eps = 1e-5
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for _, data in enumerate(train_loader):  # _是序号，data是图像内容,读取出来的图像还要再加上一维即batchsize，最终是4维
        mean += torch.mean(data['rgb'], dim=[0, 2, 3])  # mean为[3,1]
        std += torch.std(data['rgb'], dim=[0, 2, 3])
    step = len(train_data) / 128  #循环次数，这里可能会有一点误差
    mean = mean / step
    std = std / step
    #mean = torch.mean(mean, dim=0)
    #std = torch.std(std, dim=0)
    return list(mean.numpy()), list(std.numpy())


#计算pose（六维）的均值和方差
def get_pose_mean(train_data):
    """
    计算数据集的均值和标准差
    """
    #print(len(train_data))
    eps = 1e-5
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=128, shuffle=False, num_workers=4,
        pin_memory=True)
    mean = torch.zeros(6)
    std = torch.zeros(6)
    for _, data in enumerate(train_loader):  # _是序号，data是图像内容,读取出来的图像还要再加上一维即batchsize，最终是4维
        #print(data.shape)
        mean += torch.mean(data['pose'], dim=0)  # mean为[6,]
        #mean1 += data
        std += torch.std(data['pose'], dim=0)
    step = len(train_data) / 128
    mean = mean / step
    std = std / step
    #mean = mean / len(train_data)
    #newList = [x / len(train_data) for x in mean1]
    #std = std / len(train_data)

    return list(mean.numpy()), list(std.numpy())


"""for data in self.dataset:
        p = data[0]
        for i in range(6):
            # 计算每一个通道的均值和标准差
            self.means[i] += p[i, :, :, :, :, :].mean()
            self.std[i] += p[i, :, :, :, :, :].std()

    self.means = np.asarray(self.means) / num_poses
    self.std = np.asarray(self.std) / num_poses

    print("{}: normMean = {}".format(type, self.means))
    print("{}: normstd = {}".format(type, self.std))"""


if __name__ == '__main__':
    #train_dataset = DatasetFromFolder(depth_dir=r'/home/zhanlei/sfmLearner/odometry_color_dataset/train/2011_10_03_drive_0027_sync/proj_depth/groundtruth/train')
    #print(get_depth_mean(train_dataset))
    #计算RGB训练集的均值和标准差
    file = '/home/zhanl/data/kitti/data_split_seg/train_val.txt'
    train_dataset = KITTIDataset(file,
                                 data_transform=to_tensor_sample,
                                 depth_type='groundtruth')
    print("扭曲图像：")  # 这里还不确定原姿态是否正确
    print(get_pose_mean(train_dataset))
    '''print("深度：")
    print(get_depth_mean(train_dataset))
    print("RGB：")
    print(get_rgb_mean(train_dataset))'''

    #dataloader = RGBDataloader(dataroot).get_mean_std()
    #计算pose训练集的均值和方差
    #train_dataset = PoseDatasetFromFolder(pose_dir=r'/home/zhanlei/sfmLearner/odometry_color_dataset/sequences/00/image_2_pose/')
    #print(get_pose_mean(train_dataset))


