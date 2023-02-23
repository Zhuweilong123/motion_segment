# pose和rgb一一对应，设置train和test文件夹
# 随意打乱每个sequences中的图片然后以7：:3的比例分配到train和test中
import numpy as np
import os
import random
import shutil


# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:18:37 2019
将数据集按7：3划分训练集与测试集
注：当训练集和数据集中生成一遍数据后不要再用同一原始数据集生成第二遍，否则数据会有重复
@author: thik
"""

# 打开放数据的文件夹
path = r'/home/zhanlei/sfmLearner/odometry_color_dataset/train/2011_10_03_drive_0027_sync/proj_depth/groundtruth'  # 原始数据集的路径
path_a = path + '/image_02'
data_a = sorted(os.listdir(path_a))

# root=path#复制原始数据路径path

c = int(len(data_a))



'''train_root_a = path + '/train/a'  # 图像→c、d
train_root_b = path + '/train/b'  # 对应的pose→c_1、d_1
test_root_a = path + '/test/a'
test_root_b = path + '/test/b'  '''

train_root = path + '/train'
test_root = path + '/test'
# 创建测试、验证文件的子文件
for i in range(c):
    qqq = os.path.exists(train_root)  # 如果path存在，返回True；如果path不存在，返回False。
    if (not qqq):
        os.makedirs(train_root)  # 这里的train为什么要加上c[i]？这样的话不就是遍历到文件了吗？因为源代码里面在当前文件夹下面还有文件夹，最终的训练集筛选要在最底层进行

    qq = os.path.exists(test_root)
    if (not qq):
        os.makedirs(test_root)


# 这里的路径的用处是什么？这里重新赋值路径和新的内容是因为，后面打乱数据时会把之前的数据data_a也打乱掉,（因为不一定有这几个路径，等到上一步确定有了这些路径之后再赋值一次？还有一点不同是上面没有加上分隔符'/'
path1 = '/home/zhanlei/sfmLearner/odometry_color_dataset/train/2011_10_03_drive_0027_sync/proj_depth/groundtruth'  # path
path_a1 = path1 + '/image_02'  # path_a


e = []

data_0 = sorted(os.listdir(path_a1))  # 读a里的照片，这里重新赋值路径和新的内容是因为，后面打乱会把之前的数据data_a也打乱掉
random.shuffle(data_0)
for dir in data_0:
    e.append(dir)


for i in range(c):  # 这里的c只包含了png并且按顺序排序
    a = e[i]  # 读取当前文件的内容


    pic_path = path_a1 + '/' + a  # 原路径下的文件


    if i < int(c * 0.7):  # 将a中的数据按7：3分到train和test中
        obj_path = train_root + '/' + a

    else:
        obj_path = test_root + '/' + a
                # print('test:', obj_path) #显示分类情况
            # print(len(data_0),len(data_0)*0.7)
    if (os.path.exists(pic_path)):
        shutil.copyfile(pic_path, obj_path) # 往train、test中复制图片


# 把一个包含了所有pose的txt文件转换成一个pose一个txt文件
def TranPoseName():
    info = {'00': [0, 4540], '01': [0, 1100], '02': [0, 4660], '03': [0, 800], '04': [0, 270], '05': [0, 2760], '06': [0, 1100], '07': [0, 1100], '08': [1100, 5170], '09': [0, 1590]}
    # info = {'00': [0, 4540]}

    for video in info.keys():
        file_name = '/home/zhanlei/sfmLearner/odometry_color_dataset/poses_cam2/{}.txt'.format(video)  # 这里的地址如果不一样的话，需要更改哦
        with open(file_name) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]  # lines是一个list包含了文件中的每一行
            for i in range(len(lines)):  # For loop allows this to work with any number of lines
                str_num = str(i)  # 数字转化为字符串
                str_six_num = str_num.zfill(6)  # 转换为6位字符串，右对齐补0
                file = open(f'/home/zhanlei/sfmLearner/odometry_color_dataset/sequences/' + video + '/image_2/' + str_six_num + '.txt', 'w')  # Same as '{str(i+1)}.json'.format() or str(i+1)+'.json'

                file.write(lines[i])

                file.close()

