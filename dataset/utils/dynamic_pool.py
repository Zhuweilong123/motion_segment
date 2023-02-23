import os
import random


# 包含插入和删除操作,这里改成路径后是str类型的数据，要怎么对比大小
def modify_dynamicpool(dynamic_images, index, delete=False):
    re = 0  # 重复系数，表示插入的索引是否在动态池中已经重复了

    if not delete:  # 进行插入操作（按顺序插入）
        for i in range(len(dynamic_images)):
            if index == dynamic_images[i]:  # 这里可能出现的问题是，index和dynamic_images元素的类型如果不一样的话，即使数字相等也会判断不相同
                re = 1
                break
        if re == 0:
            dynamic_images.append(index)

    else:
        dynamic_images.remove(index)  # 删除这个索引
    #dynamic_images = sorted(dynamic_images, key=lambda x: str(x))  # 1.直接排序的话，python3会报错 2. 不知道为什么这里的排序结果没有返回到原列表中


# 找到动态集中最近的帧
def find_nearest_idx(number, dlist):  # mylist为128x1的张量（这里张量也可以和列表数据进行计算），dlist就是动态数据池
    original = []
    list_sorted = []
    for j in dlist:
        original.append(abs(number-j))
        list_sorted.append(abs(number-j))
    list_sorted.sort()
    #排除动态池中已有对应的帧的情况，还有一个可能的结果是正负样本是同一个
    if list_sorted[0] != 0:
        if list_sorted[0] == list_sorted[1]:
            nearest = original.index(list_sorted[0])
            nearest_2 = original.index(list_sorted[1], original.index(list_sorted[0]) + 1)
        else:
            nearest = original.index(list_sorted[0])  # 当数值一样的时候会返回对应的第一个下标
            nearest_2 = original.index(list_sorted[1])  # 得到的是一个
    elif list_sorted[1] == list_sorted[2]:
        nearest = original.index(list_sorted[1])
        nearest_2 = original.index(list_sorted[2], original.index(list_sorted[1]) + 1)
    else:
        nearest = original.index(list_sorted[1])
        nearest_2 = original.index(list_sorted[2])
    return nearest, nearest_2  # 返回的是两个列表，存储的是动态池中索引下标，nearest是128x1的列表，表示128个数据对应的正样本，nearest_2是128x1的列表，表示128个数据对应的负样本


# 随机读取10个帧作为初始动态帧
def initial_list(file_list, dy):
    sample_size = 10
    with open(file_list, "r") as f:
        data = f.readlines()
    random.shuffle(data)  # 打乱列表中的数据
    for _, fname in enumerate(data):
        depth_file = fname.split()[0].replace('image_02/data', 'proj_depth/groundtruth/image_02')
        depth_file = depth_file.replace('raw', 'depth')
        if sample_size > 0 and os.path.exists(depth_file):
            dy.append(fname.split()[0])
            dy.sort()
            sample_size -= 1

"""
# 这里相当于维持了两个数据集，实际读取的时候花费的时长应该非常高，可以尝试动态数据池继承总数据集的内容？
class DynamicDataset(data.Dataset):
    def __int__(self, image_dir, depth_dir, dynamic_images):  # dynamic_images是包含了动态帧索引号的每次迭代一次就会实时更新的列表list
        super(DynamicDataset, self).__int__()
        self.root = image_dir
        self.root_d = depth_dir
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

        self.min = int(os.path.splitext(min(self.img_filenames))[0])
        self.max = int(os.path.splitext(max(self.img_filenames))[0])
        self.pmean = [22.318, -7.904, 230.700, 0.392, -0.015, 0.010]  # pose训练集的均值
        self.pstd = [142.014, 6.214, 131.488, 1.700, 0.032, 0.042]  # pose训练集的方差

    def __getitem__(self, item):
        name = self.dynamic_index[item]  # 索引
        filenames = name + ".png"
        d = Image.open(join(self.root_d, filenames[
            item]))  # 原深度1通道读取，问题是都没有超过100的元素值，不能调用transform里的totensor函数和normalize  It-1
        d = self.transform1(d)

        im = Image.open(join(self.root, filenames[item]))
        img = self.transform2(im)  # 3,224,224

        p = open(join(self.root, name + '.txt'))
        p = p.readlines()
        p = np.array([float(i) for i in p[0].strip('\n').split(' ')])
        p = (p - self.pmean) / self.pstd
        return name, img, d, p"""
