import numpy as np
import torch
import random
import os
import torch.utils.data as data

from tqdm import tqdm
from torchvision.transforms import transforms
from visdom import Visdom

from dataset.original_method.kitti_dataset_utils import \
    pose_from_oxts_packet, read_calib_file, transform_from_rot_trans
from dataset.utils.image import load_image

from dataset.data_prepocess.Matrix_Angle import R_to_angle
from models.similarity.network.CrossModalAttention import CrossModalAttention

torch.set_default_tensor_type(torch.FloatTensor)
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("using {} device.".format(device))

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
    # print(index)
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
    return np.expand_dims(depth, axis=2)

def transform_depth(depth, tensor_type='torch.FloatTensor'):
    transform2 = transforms.Compose([  # 深度图像预处理
        transforms.ToTensor(),
        transforms.Resize([224, 68]),
        transforms.Pad([78, 0]),
        # transforms.Normalize((0.501,), (3.580,)),  # 这里均值和标准差是通过计算整个训练集图像所得
        transforms.Normalize((0.473,), (3.468,)),
    ])
    depth = transform2(depth).type(tensor_type)
    return depth

def transform_pose(p, tensor_type='torch.FloatTensor'):
    # pmean = torch.tensor([79.677, 59.714, 4.719, -0.002, -0.008, 0.352])  # pose训练集的均值
    # pstd = torch.tensor([344.457, 243.536, 19.442, 0.032, 0.033, 1.815])  # pose训练集的方差
    pmean = torch.tensor([52.962, 28.659, -0.100, -0.002, -0.006, 0.349])  # pose训练集的均值
    pstd = torch.tensor([29.689, 27.463, 1.266, 0.013, 0.013, 0.4444])  # pose训练集的方差
    return ((p - pmean) / pstd).type(tensor_type)

########################################################################################################################
#### DATASET
########################################################################################################################

class testset(data.Dataset):
    # with_dynamic代表是否有对应的动态帧，dynamic_list包含了训练集中所有动态帧的路径
    def __init__(self, file, depth_type='groundtruth'):
        # backward_context=0, forward_context=0, strides=(1,)
        # Assertions

        self.split = file.split('/')[-1].split('.')[0]
        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None  # 即depth_type非空即为True


        # 这里的cache的理解：保存了上一个数据的读取记录
        self._cache = {}  # 存储的是最后以个文件夹路径的文件数量（每个序列参照深度图数量）
        self.pose_cache = {}
        self.oxts_cache = {}
        self.calibration_cache = {}
        self.imu2velo_calib_cache = {}
        self.sequence_origin_cache = {}

        with open(file, "r") as f:
            data = f.readlines()  # data是列表类型的数据，一行代表一个元素

        self.paths = []
        self.labels = []  # 和self.path一一对应，记录了对应路径帧的动静态类别

        for i, fname in enumerate(data):

            path = fname.split()[0]
            label = fname.split()[-1]

            if os.path.exists(path):
                if not self.with_depth:  # 不需要深度信息
                    self.paths.append(path)
                    self.labels.append(label)
                else:  # 如果要输入深度图片的话，再添加一个深度路径

                    depth = self._get_depth_file(path)  # depth是该rgb图片对应的深度图路径，但也有可能找不到对应的深度图
                    if depth is not None and os.path.exists(
                            depth):  # 如果找不到当前rgb对应的depth即不把该rgb加入到路径中，即解决了深度图要比rgb图少十帧的问题
                        self.paths.append(path)
                        self.labels.append(label)

    ####################################################################################################################
    # @为python装饰器：可以让某个函数在不改动代码的基础上增加额外的功能。比如函数的嵌套：staticmethod(func)
    @staticmethod
    def _get_next_file(idx, file):
        # 返回的是当前文件路径下往后数idx帧的路径
        """Get next file given next idx and current file."""
        base, ext = os.path.splitext(os.path.basename(file))  # base=0000000000,ext=.png
        return os.path.join(os.path.dirname(file), str(idx + int(base)).zfill(len(base)) + ext)  # zfill是向右对齐添零

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
        origin_frame = os.path.join(os.path.dirname(image_file),
                                    str(0).zfill(len(base)) + ext)  # 每个时间序列的0000000000.png

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
        pose = pose.to(torch.float32)  # 得到的是六自由度姿态：x、y、z方向平移距离和角度偏移值
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

        c_f = self.paths[idx]  # 当前路径
        label = int(self.labels[idx])  # 字符型转成int型
        depth = transform_depth(self._read_depth(self._get_depth_file(c_f)))
        pose = transform_pose(self._get_pose(c_f))

        l_f = self._get_next_file(-1, c_f)
        n_f = self._get_next_file(1, c_f)

        # 判断是否前一帧和后一帧都存在
        l_f, n_f = self.judge_adjacent(l_f, n_f)
        depth_l = transform_depth(self._read_depth(self._get_depth_file(l_f)))
        pose_l = transform_pose(self._get_pose(l_f))

        depth_n = transform_depth(self._read_depth(self._get_depth_file(n_f)))
        pose_n = transform_pose(self._get_pose(n_f))

        # 随机取一静态帧作为正样本（这里要保证和当前帧不是一帧），这里怎么保证每次loader取到的都是不一样的？
        positive = random.choice([x for x in self.paths if x != c_f])

        depth_p = transform_depth(self._read_depth(self._get_depth_file(positive)))
        pose_p = transform_pose(self._get_pose(positive))

        # index_cp = extract_idx(c_f) - extract_idx(positive)
        label_p = int(self.labels[self.paths.index(positive)])  # 判断该帧所属标签
        if label_p != label:
            diff = 1  # 不是同类帧
        else:
            diff = 0  # 是同类帧

        dp = torch.cat((depth, depth_p), dim=0)  # 和任意一个静态帧
        pp = torch.cat((pose, pose_p), dim=0)

        # 验证的时候需要计算相邻帧：

        # 如果相邻两帧和当前帧的距离都大于动静帧的margin则认为当前帧是动态帧，否则是静态帧：
        dl = torch.cat((depth, depth_l), dim=0)
        pl = torch.cat((pose, pose_l), dim=0)

        dn = torch.cat((depth, depth_n), dim=0)
        pn = torch.cat((pose, pose_n), dim=0)



        # return p1, d1, p2, d2, pp, dp, pn, dn, label
        # 输出的格式是当前帧和任意一帧的concat姿态深度、当前帧和前后相邻帧的concat姿态深度、当前帧的标签、当前帧和所选帧的标签一致性
        return pp, dp, pl, dl, pn, dn, label, diff


########################################################################################################################
if __name__ == "__main__":
    root = '/home/zhanl/data/kitti/data_splits/label/test_shuffle.txt'
    checkpoint_dir = "/home/zhanl/data/code/encoder/models/encoder/checkpoint/net_epoch_10_best.pth"
    test_dataset = testset(file=root, depth_type='groundtruth')  # resize的时候改成等比例缩放
    test_num = len(test_dataset)
    test_loader = data.DataLoader(
        test_dataset, batch_size=128,
        shuffle=True, num_workers=4,
        # collate_fn=lambda x:x,
        pin_memory=True)
    test_bar = tqdm(test_loader)
    print("---Loading_Model---")
    net = CrossModalAttention()
    net = net.type(torch.FloatTensor)
    net = torch.load(checkpoint_dir, map_location=device)

    # 可视化visdom类
    vis = Visdom(env='test')
    vis.line([[0., 0.]], [0.],  # Y坐标的起点和X坐标的起点
             win='test accuracy',  # 窗口名称
             opts=dict(title="test_acc", xlabel="step", ylabel="acc", legend=['adjacent acc', 'class acc']))  # 标题和横坐标纵坐标名称


    threshold = 0.2
    margin = 0.8
    for step, data in enumerate(test_bar):
        batch = data[0].shape[0]  # 一个batch里面的数据数量
        label = data[6].to(device)  # 全部静态帧

        # threshold = (args.margin + args.Margin)/2
        positive = net(data[0].to(device), data[1].to(device))  # 当前帧和任意帧的距离
        last = net(data[2].to(device), data[3].to(device))
        next = net(data[4].to(device), data[5].to(device))

        s1 = last.lt(threshold).float() + next.lt(threshold).float()  # 前后帧的距离都小于该值即相异度，则证明其是静态帧
        tmp = torch.full([batch, 1], 2).to(device)  # 创建一个值全部为2的128x1维张量
        pred = torch.eq(s1, tmp).float().to(device)  # 只保留两个值都为1的下标，预测其为静态帧，记录的值为1
        correct1 = torch.eq(pred, label).float().sum().item()  # 这里记录为1的为静态帧即为预测成功的

        s2 = positive.gt(margin).float()  # 距离大于0.8肯定就是属于不同类：是1
        diff = data[7].to(device)  # 判断当前批次中的任意两帧是否是同类或者异类
        correct2 = torch.eq(diff, s2).float().sum().item()

        adj_acc = correct1/batch
        class_acc = correct2/batch
        # 输出预测和真实标签相同的图片数量
        # correct = torch.eq(pred, label).float().sum().item()  # .float()将truefalse转成1和0，.sum()将所有值加起来，还是tensor型数据，.item()可以得到值
        vis.line([[adj_acc, class_acc]], [step], win='test accuracy', update='append')




