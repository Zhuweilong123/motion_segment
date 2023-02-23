# netron读取本地模型
import os

from netron import start
import numpy as np
import tqdm
from path import Path
from numpy import load
import cv2
import torch

from utils.loss import seg_loss

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '/home/zhanl/data/kitti/raw/2011_09_26/2011_09_26_drive_0014_sync/mask/np_array_All_classes_Output/0000000038.npz'
with np.load(path) as data:
    mask = data['x']
    print(np.max(mask))
    np.savetxt('./38.txt', mask.reshape(750, -1))


root = '/home/zhanl/data/kitti/raw/2011_09_26/'
dir1 = sorted(os.listdir(root))
mask_path = '/mask/np_array_All_classes_Output/'

for i in dir1:
    path = root + i + mask_path
    dir2 = sorted(os.listdir(path))
    for j in dir2:
        path2 = path + j
        with np.load(path2) as data:
            mask = data['x']  # [H,W,C] C是instant的个数，即识别出几个物体
            if mask.shape[-1] != 2484:  # 图片不存在所给类别的标签时，shape为[h,w]不需要再处理
                new_mask = np.argmax(mask, axis=-1)
                max1 = np.amax(mask, axis=-1)
                if np.max(max1) > 8:
                    print(path2)
                arr = np.sum(mask, axis=-1) > 8
                #max1[arr] =
                channel_index = np.where(np.amax(mask, axis=(0, 1)) == 8)
                # 删除最大值为8的那个通道
                mask[:, :, channel_index] = 0
                # mask2 = np.delete(mask, channel_index, axis=-1)
                # 即存在最大类别即misc，需要把这个维度给删除
                mask = np.sum(mask, axis=-1)  # 每个像素的像素值记录的是该像素所属类别
                '''if np.max(mask) >= 8:
                    print(np.max(mask))
                    print(path)'''


num_classes = 9
with load(path) as data:
    mask = data['x']  # [H,W,C] C是instant的个数，即识别出几个物体
    height, width = mask.shape[0], mask.shape[1]
    if mask.shape[-1] == 2484:  # 图片不存在所给类别的标签时，shape为[h,w]
        num_instances = 0
    else:
        num_instances = mask.shape[-1]
    new_label = np.zeros((height, width, num_classes), dtype=np.float32)
    # 实例级别→像素级别，0和8怎么办
    for j in range(num_instances):
        # 每个instance通道的标签
        instance_label = mask[:, :, j]
        # 逐一计算每个通道
        class_indices = np.unique(instance_label)  # 记录了0和number
        class_indices = int(class_indices[class_indices != 0][0])  # 0表示背景需要删除，并读取其中唯一的元素,TODO 本来就只有0的怎么办
        new_label[:, :, class_indices] += instance_label

data_root = '/home/zhanl/data/kitti/raw/'
sequence_set = '2011_09_26/2011_09_26_drive_0001_sync'
#sequence_set = '09'
data_root = Path(data_root)
im_sequences = []
poses_sequences = []
indices_sequences = []
seq_length = 3
step = 1
demi_length = (seq_length - 1) // 2
shift_range = np.array([step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)  #(-1, 2)

sequences = set()
#set是集合的一种，这里测试的序列还包含了该序号之前的所有数据
#for seq in sequence_set:
    #corresponding_dirs = set((data_root + sequence_set).dirs(seq))

    #sequences = sequences | corresponding_dirs #联合操作，即dirs也包含在里面

sequence = data_root + sequence_set
#print('getting test metadata for theses sequences : {}'.format(sequences))
#for sequence in tqdm(sequences): #进度条读取
    #poses = np.genfromtxt(data_root/'poses'/'{}.txt'.format(sequence.name)).astype(np.float64).reshape(-1, 3, 4)
    #imgs = sorted((sequence/'image_2').files('*.png'))
imgs = sorted((sequence/'image_02/data').files('*.png'))
# construct 5-snippet sequences
tgt_indices = np.arange(demi_length, len(imgs) - demi_length).reshape(-1, 1)
snippet_indices = shift_range + tgt_indices
im_sequences.append(imgs)
#poses_sequences.append(poses)
indices_sequences.append(snippet_indices)
#return im_sequences, indices_sequences

root = '/home/zhanl/data/kitti/raw/2011_09_26/2011_09_26_drive_0001_sync/mask/np_array_All_classes_Output/'
dir = sorted(os.listdir(root))
palette = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]  # 每个类别在原mask里的数值
with load(root + dir[72]) as data:
    npData = data['x']
    # npData = npData.reshape(-1, 750, 2484)
    # npData = npData.sum(axis=2)  # instance转换成all-class

# image_root = '/home/zhanl/data/kitti/raw/2011_09_26/2011_09_26_drive_0001_sync/mask/binary_img_Output/'
# dir2 = sorted(os.listdir(image_root))
# img = cv2.imread(image_root+dir2[61], cv2.IMREAD_GRAYSCALE)


with open("file1.txt", 'w') as data1:
    for d in npData:
        np.savetxt(data1, d, fmt='%f', delimiter=',')

# print(np.shape(npData))


modelData = "/home/zhanl/data/code/encoder/models/encoder/checkpoint/best.onnx"  # 定义模型数据保存的路径
start(modelData)  # 输出网络结构
'''

# 在线将模型转成onnx格式
import torch
import torch.onnx
from torch.autograd import Variable
import onnx
from onnx import shape_inference
from models.encoder.ResNet import CrossModalAttention
if __name__ == '__main__':
    net = CrossModalAttention()
    #其中ResidualAttentionModel_448input()为自己搭建的网络结构
    net.cuda()

    print('net: {}'.format(net))

    path = "/home/zhanl/data/code/encoder/models/encoder/checkpoint/"
    model = path+'best.onnx'
    # torch.save(net, model)
    # model = torch.load(model)  # 加载

    x = Variable(torch.randn(1, 12)).cuda()
    y = Variable(torch.randn(1, 2, 224, 224)).cuda()
    torch_out = torch.onnx.export(net, (x, y), model, export_params=True, opset_version=11)  # 模型转换为onnx格式

    onnx.save(onnx.shape_inference.infer_shapes(onnx.load(model)), model)  # 保存模型结构的细节信息
    '''
