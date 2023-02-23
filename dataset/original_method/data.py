from turtle import pd

import torch
import torch.nn as nn
import os
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import transforms
import numpy as np
import cv2
import torch.utils.data as data
from numpy import average, linalg, dot

# 将16位的图像转换成8位的图像，使用的方法是线性缩放，这里本来是用在补全的深度图上，现在暂时用不上
def transfer16_8(img):
    img_min = np.min(img)
    img_max = np.max(img)
    img_8bit = np.array(np.rint(255 * ((img - img_min) / (img_max - img_min))), dtype=np.uint8)
    return img_8bit


def transfer16_01(img):  # uint16位的图像转到(0,1)区间内
    img_01 = img/(255**2)
    return img_01


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


# 包含了所有的帧，不分动态和静态，实例化参数分别是rgb路径和深度图姿态路径，输出是图片索引（文件前缀），经过归一化的rgb图像、深度图和姿态
class KittiDataset(data.Dataset):
    def __init__(self, image_dir, depth_dir, pose_dir):
        super(KittiDataset, self).__init__()
        self.root = image_dir
        self.root_d = depth_dir
        self.root_p = pose_dir


        # 加上排序函数sorted 文件的读取才是按照顺序进行的
        self.img_filenames = [x for x in sorted(listdir(self.root_d)) if is_image_file(x)]

        self.transform1 = transforms.Compose([  # 深度图像预处理
            #transforms.ToPILImage(),  # 这里又会自动转变为RGB图像
            #transforms.Grayscale(1),  转换成灰度图像
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # 将其先由HWC转置为CHW格式，再转为float后每个像素除以255：和kitti深度图像预处理操作一样，这里有一个问题是只对uint8才除以255，所以这里的操作仅为由HWC转置为CHW格式
            transfer16_01,
            transforms.Normalize((0.082,), (0.133,)),  # 这里均值和标准差是通过计算整个训练集图像所得
        ])
        self.transform2 = transforms.Compose([  # RGB图像预处理
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.409, 0.4334, 0.441], std=[0.288, 0.300, 0.311]),
        ])
        self.transform3 = transforms.Compose([  # RGB图像预处理计算视觉相似度
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.min = int(os.path.splitext(min(self.img_filenames))[0])
        self.max = int(os.path.splitext(max(self.img_filenames))[0])
        self.pmean = [22.318, -7.904, 230.700, 0.392, -0.015, 0.010]  # pose训练集的均值
        self.pstd = [142.014, 6.214, 131.488, 1.700, 0.032, 0.042]  # pose训练集的方差
        dir_path = os.path.dirname(os.path.realpath(__file__))

        test_scene_file = os.path.join(dir_path, 'splits/test_scene_eigen.txt')
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]

        self.dataset_dir = "/home/zhanlei/data/train/"
        #self.cam_ids = ['02']
        self.seq_list = ['00', '01', '02', '05', '06', '07', '08', '09']
        self.mode = ['image_2_depth', 'image_2', 'image_2_pose']  # 数据的类型分成三种，rgb、depth和pose
        self.use_gt = False

        self.collect_train_frames()


    def collect_train_frames(self):
        all_frames = []
        all_depths = []
        all_poses = []
        for seq in self.seq_list:
            seq_set = os.listdir(os.path.join(self.dataset_dir, seq))  # 00~09序列下的文件夹名
            depth_dir = os.path.join(self.dataset_dir, seq, self.mode[0])
            #if dr[:-5] in self.test_scenes:
                #continue
            #for cam in self.cam_ids:
            if os.path.exists(depth_dir):
                depths = sorted(os.listdir(depth_dir))
                for depth in depths:
                    all_depths.append(seq + ' ' + self.mode[0] + ' ' + depth)  # 包含了所有深度图的路径
                    all_frames.append(seq + ' ' + self.mode[1] + ' ' + depth)
                    #all_poses.append(seq + ' ' + self.mode[2] + ' ' + os.)

        self.train_frames = all_frames
        self.train_depths = all_depths

    def load_single_example(self, frames, tgt_idx):
        tgt_drive, tgt_cid, tgt_frame_id = frames[tgt_idx].split(' ')
        tgt_date = tgt_drive[:10]
        raw_image = self.load_image_raw(tgt_date, tgt_drive, tgt_cid, tgt_frame_id)
        zoom_y = self.img_height / raw_image.size[0]
        zoom_x = self.img_width / raw_image.size[1]
        intrinsics = self.load_intrinsics_raw(tgt_date, tgt_cid)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        cur_img = raw_image.resize((self.img_height, self.img_width), Image.NEAREST)

        raw_dmap = self.load_gt_raw(tgt_drive, tgt_frame_id)
        cur_dmap = raw_dmap.resize((self.img_height, self.img_width), Image.NEAREST)

        example = {}
        example['image'] = cur_img
        example['dmap'] = cur_dmap
        example['intrinsics'] = intrinsics
        example['folder_name'] = tgt_drive + '_' + tgt_cid + '/'
        example['file_name'] = tgt_frame_id

        return example

    def load_gt_raw(self, tgt_drive, tgt_frame_id):
        dmap_path = '{}/{}/{}/proj_depth/groundtruth/image_02/{}'. \
            format(self.dataset_dir, self.mode, tgt_drive, tgt_frame_id)
        dmap_raw = Image.open(dmap_path).convert('RGB')

        return dmap_raw

    def load_image_raw(self, date, drive, cam_id, frame):
        # data: 2011_09_26
        # drive: 2011_09_26_XXX
        # cam_id: 02/03
        # frame: 00000005
        im_path = os.path.join(self.dataset_dir, 'rawdata', date, drive, 'image_{}/data/'.format(cam_id), frame)
        im = Image.open(im_path).convert('RGB')
        return im

    def load_intrinsics_raw(self, date, cid):
        calib_file = os.path.join(self.dataset_dir, 'rawdata', date, 'calib_cam_to_cam.txt')

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
        intrinsics = P_rect[:3, :3]
        return intrinsics

    def read_raw_calib_file(self, filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
            return data

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0, 0] *= sx
        out[0, 2] *= sx
        out[1, 1] *= sy
        out[1, 2] *= sy
        return


    def __getitem__(self, item):

        #for seq in self.seq_list:

        name = os.path.splitext(self.img_filenames[item])[0]  # 可作为第一张图像真正的索引,数据类型是str

        d = Image.open(join(self.root_d, self.img_filenames[item]))  # 原深度1通道读取，问题是都没有超过100的元素值，不能调用transform里的totensor函数和normalize  It-1
        d = self.transform1(d)
        # d = torch.stack((d1, d2), axis=0)
        # d = d.view([2, 224, 224])  # 两个通道的深度图

        im = Image.open(join(self.root, self.img_filenames[item]))
        img = self.transform2(im)  # 3,224,224

        p = open(join(self.root_p, name + '.txt'))
        p = p.readlines()
        p = np.array([float(i) for i in p[0].strip('\n').split(' ')])
        p = (p - self.pmean) / self.pstd
        #p = np.vstack((p1, p2))  # 竖直拼接numpy数组
        #p = torch.from_numpy(np.float32(p))  # numpy要转换成12维的tensor

        return name, img, d, p

    def __len__(self):  # 获取数据集的长度
        return len(self.img_filenames)  # 这里可能规定了item的范围是(0,len-1),因为最后一个索引找不到对应的洗一个作对比


# 这里相当于维持了两个数据集，实际读取的时候花费的时长应该非常高，可以尝试动态数据池继承总数据集的内容？
class DynamicDataset(data.Dataset):
    def __init__(self, image_dir, depth_dir, pose_dir, dynamic_images):  # dynamic_images是包含了动态帧索引号的每次迭代一次就会实时更新的列表list
        super(DynamicDataset, self).__init__()
        self.root = image_dir
        self.root_d = depth_dir
        self.root_p = pose_dir
        self.dynamic_index = dynamic_images

        self.transform1 = transforms.Compose([  # 深度图像预处理
            # transforms.ToPILImage(),  # 这里又会自动转变为RGB图像
            # transforms.Grayscale(1),  转换成灰度图像
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # 将其先由HWC转置为CHW格式，再转为float后每个像素除以255：和kitti深度图像预处理操作一样，这里有一个问题是只对uint8才除以255，所以这里的操作仅为由HWC转置为CHW格式
            transfer16_01,
            transforms.Normalize((0.082,), (0.133,)),  # 这里均值和标准差是通过计算整个训练集图像所得
        ])
        self.transform2 = transforms.Compose([  # RGB图像预处理
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.409, 0.4334, 0.441], std=[0.288, 0.300, 0.311]),
        ])
        self.pmean = [22.318, -7.904, 230.700, 0.392, -0.015, 0.010]  # pose训练集的均值
        self.pstd = [142.014, 6.214, 131.488, 1.700, 0.032, 0.042]  # pose训练集的方差

    def __getitem__(self, item):
        name = self.dynamic_index[item]  # 索引，这里的name是元组型的数据，因为动态池列表中存储的即为一个一个的元组数据
        name = "".join(tuple(name))  # 元组数据转换成字符型

        d = Image.open(join(self.root_d, name + '.png'))  # 原深度1通道读取，问题是都没有超过100的元素值，不能调用transform里的totensor函数和normalize  It-1
        d = self.transform1(d)

        im = Image.open(join(self.root, name + '.png'))
        img = self.transform2(im)  # 3,224,224

        p = open(join(self.root_p, name + '.txt'))
        p = p.readlines()
        p = np.array([float(i) for i in p[0].strip('\n').split(' ')])
        p = (p - self.pmean) / self.pstd
        return name, img, d, p

    def __len__(self):
        return len(self.dynamic_index)



"""
# 基础数据集，输入rgb路径、深度姿态路径，返回拼接好的深度和姿态数组（归一化后）
class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, depth_dir):
        super(DatasetFromFolder, self).__init__()
        self.root = image_dir
        self.root_d = depth_dir

        self.img_filenames = [x for x in listdir(self.root_d) if is_image_file(x)]

        self.transform1 = transforms.Compose([  # 深度图像预处理
            #transforms.ToPILImage(),  # 这里又会自动转变为RGB图像
            #transforms.Grayscale(1),  转换成灰度图像
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),  # 将其先由HWC转置为CHW格式，再转为float后每个像素除以255：和kitti深度图像预处理操作一样，这里有一个问题是只对uint8才除以255，所以这里的操作仅为由HWC转置为CHW格式
            transfer16_01,
            transforms.Normalize((0.082,), (0.133,)),  # 这里均值和标准差是通过计算整个训练集图像所得
        ])
        self.transform2 = transforms.Compose([  # RGB图像预处理
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.409, 0.4334, 0.441], std=[0.288, 0.300, 0.311]),
        ])
        self.transform3 = transforms.Compose([  # RGB图像预处理计算视觉相似度
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.min = int(os.path.splitext(min(self.img_filenames))[0])
        self.max = int(os.path.splitext(max(self.img_filenames))[0])
        self.pmean = [22.318, -7.904, 230.700, 0.392, -0.015, 0.010]  # pose训练集的均值
        self.pstd = [142.014, 6.214, 131.488, 1.700, 0.032, 0.042]  # pose训练集的方差

    def __getitem__(self, item):

        #  计算索引差值并转换到（0,1）区间中

        name1 = os.path.splitext(self.img_filenames[item])[0]  # 可作为第一张图像真正的索引
        name2 = os.path.splitext(self.img_filenames[item+1])[0]  # 作为第二张图像真正的索引
        diff = abs(int(name2)-int(name1))  # 二者时间戳之间的差值
        diff = 1 - (diff-self.min)/(self.max-self.min)

        d1 = Image.open(join(self.root_d, self.img_filenames[item]))  # 原深度1通道读取，问题是都没有超过100的元素值，不能调用transform里的totensor函数和normalize  It-1

        #im1 = cv2.imread(join(self.root, self.img_filenames[item]))
        im1 = Image.open(join(self.root, self.img_filenames[item]))
        p1 = open(join(self.root, name1 + '.txt'))
        p1 = p1.readlines()
        p1 = np.array([float(i) for i in p1[0].strip('\n').split(' ')])
        p1 = (p1 - self.pmean) / self.pstd

        d2 = Image.open(join(self.root_d, self.img_filenames[item+1]))  #

        im2 = cv2.imread(join(self.root, self.img_filenames[item+1]))
        im2 = Image.open(join(self.root, self.img_filenames[item+1]))
        #cosin = image_similarity_vectors_via_numpy(im1, im2)  # 两张RGB图像的余弦相似度

        p2 = open(join(self.root, name2 + '.txt'))
        p2 = p2.readlines()
        p2 = np.array([float(i) for i in p2[0].strip('\n').split(' ')])
        p2 = (p2 - self.pmean)/self.pstd

        d1 = self.transform1(d1)
        d2 = self.transform1(d2)
        d = torch.stack((d1, d2), axis=0)
        d = d.view([2, 224, 224])  # 两个通道的深度图

        im1 = np.array(im1)
        im2 = np.array(im2)
        im = np.concatenate((im1, im2), axis=0)  # 垂直拼接
        im = self.transform2(im)
        img1 = self.transform3(im1)  # 3,224,224
        img2 = self.transform3(im2)  # 3,224,224
        #cos = F.cosine_similarity(img1.reshape(1, -1), img2.reshape(1, -1), dim=1, eps=1e-08)

        p = np.vstack((p1, p2))  # 竖直拼接numpy数组
        p = torch.from_numpy(np.float32(p))  # numpy要转换成12维的tensor
        return d, p

    def __len__(self):  # 获取数据集的长度
        return len(self.img_filenames)-1  # 这里可能规定了item的范围是(0,len-1),因为最后一个索引找不到对应的洗一个作对比


# RGB数据集
class RGBdatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(RGBdatasetFromFolder, self).__init__()
        self.root = image_dir
        self.img_filenames = [x for x in sorted(listdir(self.root)) if is_image_file(x)]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __getitem__(self, item):
        im1 = cv2.imread(join(self.root, self.image_filenames[item+5]))  # 对应的rgb图片序号从05开始
        im2 = cv2.imread(join(self.root, self.image_filenames[item+6]))
        t = np.concatenate((im1, im2), axis=0)  # 垂直拼接
        t = self.transform(t)
        return t

    def __len__(self):  # 获取数据集的长度
        return len(self.image_filenames)
"""