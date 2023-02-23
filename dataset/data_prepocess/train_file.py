import os
import numpy as np

all = []
root = '/home/zhanl/data/kitti/raw/2011_09_26/'
dir = sorted(os.listdir(root))
dir1 = dir[:33]  # 只取有标注的图片作为训练集:1-93,一共12919张
dir2 = dir[33:38]
#np.random.shuffle(dir1)  # 为了计算出两个通道的均值和方差，大范围是顺序没有打乱的，即train在val前再在test前
train_val_file = '/home/zhanl/data/code/motion_seg/data/train_val.txt'
train_file = '/home/zhanl/data/code/motion_seg/data/train.txt'
val_file = '/home/zhanl/data/code/motion_seg/data/val.txt'
test_file = '/home/zhanl/data/code/motion_seg/data/test.txt'
image_path = '/image_02/data/'

for i in dir1:
    path = root + i + image_path
    dir_1 = os.listdir(path)
    for j in dir_1:
        # 去掉每个文件下的第一张图片，最后一共是12881张
        if j == '0000000000.png':
            continue
        all.append(path+j)
np.random.shuffle(all)

test = []
# 84-93
for i in dir2:
    path = root + i + image_path
    dir_2 = os.listdir(path)
    for j in dir_2:
        # 去掉每个文件下的第一张图片，最后一共是12881张
        if j == '0000000000.png':
            continue
        test.append(path+j)

# 计算均值方差
train_val = all  # 10304张→10295张
# K折交叉验证
train = train_val[:int(len(train_val)*(3/4))]  # 7728张→7721张
val = train_val[int(len(train_val)*(3/4)):]  # 2576张→2584张

#test = all[int(len(all)*0.8):]  # 2577张→2586张

np.random.shuffle(train)
np.random.shuffle(val)
np.random.shuffle(test)
np.random.shuffle(train_val)

with open(train_file, 'w') as f:
    for i in train:
        f.write(i + '\n')

with open(val_file, 'w') as f:
    for i in val:
        f.write(i + '\n')

with open(test_file, 'w') as f:
    for i in test:
        f.write(i + '\n')

with open(train_val_file, 'w') as f:
    for i in train_val:
        f.write(i + '\n')


# 构造时间戳和灰度图路径拼接的txt文件
timastamp = []
root = '/home/zhanl/data/kitti/raw/2011_09_26/2011_09_26_drive_0009_sync/image_00/'
dirs = sorted(os.listdir(root+'data/'))
with open(root+'timestamps.txt', 'r') as f:
    data0 = f.readlines()
for i, lines in enumerate(data0):
    times = lines.split(":")[-1].split("\n")[0]
    path = 'data/' + dirs[i]
    timastamp.append(times+' '+path)

file = '/home/zhanl/data/kitti/raw/2011_09_26/2011_09_26_drive_0009_sync/image_00/associate.txt'
with open(file, "w") as f1:
    for j in timastamp:
        f1.write(j+'\n')



'''
train_index = []
label_index = []
train_list = '/home/zhanl/data/kitti/data_splits/dynamic/dynamic_train.txt'
with open(train_list, 'r') as f:
    data1 = f.readlines()
for i, fpath in enumerate(data1):
    path = fpath.split()[0]
    #label = fpath.split()[-1]
    train_index.append(path)

np.random.shuffle(train_index)
train = train_index[:int(len(train_index)*(2/3))]
val = train_index[int(len(train_index)*(2/3)):]
np.random.shuffle(train)
np.random.shuffle(val)


with open('/home/zhanl/data/kitti/data_splits/label/dynamic/train.txt', 'w') as f:
    for i in train:
        f.write(i + '\n')

with open('/home/zhanl/data/kitti/data_splits/label/dynamic/val.txt', 'w') as f:
    for i in val:
        f.write(i + '\n')



dynamic_index = []
train_index = []
#包含了所有动态帧路径的列表
train_list = '/home/zhanl/data/kitti/data_splits/test.txt'
train1_list = '/home/zhanl/data/kitti/data_splits/test_label.txt'
dynamic_list = '/home/zhanl/data/kitti/data_splits/dynamic/dynamic_list.txt'
with open(dynamic_list, 'r') as f:
    data = f.readlines()
for i, fpath in enumerate(data):
    path = fpath.split()[0]
    dynamic_index.append(path)
#计算训练集图片数据
with open(train_list, 'r') as f:
    data1 = f.readlines()
for i, fpath in enumerate(data1):
    path = fpath.split()[0]
    train_index.append(path)


imgs_training_length = len(train_index)
fileTrain = open(train1_list, 'w')
#通过for循环，将训练集图片路径，和标签组成字符串，存储到文件training.txt中
for ip in range(imgs_training_length):
    if train_index[ip] in dynamic_index:
        label = 1
    else:
        label = 0
    #使用str（）内部函数，将标签的整型变量转变成字符串，并与训练集图片路径拼接
    temp_data = train_index[ip]+'    '+str(label)
    fileTrain.write(temp_data)
    fileTrain.write('\n')
fileTrain.close()
filename = []
test_name = []
#label = []
dirs = sorted(os.listdir(root))  # 2011_09_26,2011_09_29,2011_09_30,2011_10_03
for dir in dirs:
    dir_path = root + '/' + dir
    seqs = sorted(os.listdir(dir_path))  # 0001_sync
    for n in seqs:
        seq_path = dir_path + '/' + n + '/' + 'image_02' + '/' + 'data'
        if os.path.isdir(seq_path):
            names = sorted(os.listdir(seq_path))
            for s in names:
                path = root + '/' + dir + '/' + n + '/' + 'image_02' + '/' + 'data' + '/' + s + '\t'
                filename.append(path)
                #idx = extract_idx(path)
                #if os.path.exists(path):1003/0027/2195.png
                #if (extract_idx(path) <= 100300270000002195):
                
                if path not in dynamic_index:
                    if (extract_idx(path) <= 100300270000002833):
                        filename.append(root + '/' + dir + '/' + n + '/' + 'image_02' + '/' + 'data' + '/' + s + '\t')
                    else:
                        test_name.append(root + '/' + dir + '/' + n + '/' + 'image_02' + '/' + 'data' + '/' + s + '\t')
'''
dynamic_list = '/home/zhanl/data/kitti/data_splits/dynamic/dynamic_train.txt'
static_list = '/home/zhanl/data/kitti/data_splits/static/static_train.txt'
train_list = '/home/zhanl/data/kitti/data_splits/train_label.txt'

filename = []
static_index = []
dynamic_index = []
#计算训练集图片数据
with open(dynamic_list, 'r') as f:
    data = f.readlines()
for i, fpath in enumerate(data):
    path = fpath.split()[0]
    dynamic_index.append(path)
with open(static_list, 'r') as f:
    data1 = f.readlines()
for i, fpath in enumerate(data1):
    path = fpath.split()[0]
    static_index.append(path)


#划分训练集、测试集，默认比例4:1
train_dynamic = dynamic_index[:int(len(dynamic_index)*0.8)]
train_static = static_index[:int(len(static_index)*0.8)]

filename = sorted(train_dynamic + train_static)
with open('/home/zhanl/data/kitti/data_splits/train_ds.txt', 'w') as f:
    for i in filename:
        f.write(i + '\n')


test_dynamic = dynamic_index[int(len(dynamic_index)*0.8):]
test_static = static_index[int(len(static_index)*0.8):]
#打乱文件名列表
#np.random.shuffle(train_dynamic)
np.random.shuffle(train_static)
np.random.shuffle(test_dynamic)
np.random.shuffle(test_static)


with open('/home/zhanl/data/kitti/data_splits/label/train.txt', 'w') as f1, open('/home/zhanl/data/kitti/data_splits/label/test.txt', 'w') as f2:
    for i in train_dynamic:
        label = 1
        f1.write(i + '    ' + str(label) + '\n')
    for i in train_static:
        label = 0
        f1.write(i + '    ' + str(label) + '\n')

    for j in test_dynamic:
        label = 1
        f2.write(j+'    '+str(label) + '\n')
    for j in test_static:
        label = 0
        f2.write(j+'    '+str(label) + '\n')
'''
lines = []
out_file = open("/home/zhanl/data/kitti/data_splits/label/train_shuffle.txt", 'w')
with open("/home/zhanl/data/kitti/data_splits/label/train.txt", 'r') as f:
    data = f.readlines()
for i, fpath in enumerate(data):
    #path = fpath.split()[0]
    lines.append(fpath)

np.random.shuffle(lines)

for line in lines:
    out_file.write(line)

out_file.close()

lines2 = []
out2_file = open("/home/zhanl/data/kitti/data_splits/label/test_shuffle.txt", 'w')
with open("/home/zhanl/data/kitti/data_splits/label/test.txt", 'r') as f:  # 需要打乱的原文件位置
    data1 = f.readlines()
    for i, fpath in enumerate(data1):
        #path = fpath.split()[0]
        lines2.append(fpath)
np.random.shuffle(lines2)

for line in lines2:
    out2_file.write(line)

out2_file.close()

print("成功")

#分别写入train.txt, test.txt



print('成功！')'''

