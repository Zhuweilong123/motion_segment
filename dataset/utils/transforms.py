# 主要任务：调整图像尺寸、调整深度尺寸、改变字典内图像和内参的尺寸、改变字典内图像，内参和深度图的尺寸
# 将字典里的关键词（rgb、depth）对应的值转成tensor、数据（字典）增强
import cv2
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image

from dataset.utils.misc import filter_dict


def resize_image(image, shape, interpolation=Image.ANTIALIAS):
    """
    Resizes input image.

    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode

    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)


def resize_depth(depth, shape):
    """
    Resizes depth map.

    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape

    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    #return np.expand_dims(depth, axis=2)
    return depth


# resize图片后内参也要resize
def resize_sample_image_and_intrinsics(image, intr, shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes the image and intrinsics of a sample

    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)  256*832
        Output shape
    image_interpolation : int
        Interpolation mode

    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = image.size
    (out_h, out_w) = shape
    # Scale intrinsics
    intrinsics = np.copy(intr)
    intrinsics[0] *= out_w / orig_w
    intrinsics[1] *= out_h / orig_h
    intr = intrinsics

    # Scale images
    image = image_transform(image)

    # Return resized sample
    return image, intr


########################################################################################################################
def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)


# 这里还有一个疑问是：是否需要scale+random crop+flip之类来进行数据增强和打乱
def transform_rgb(image, tensor_type='torch.FloatTensor'):
    transform1 = transforms.Compose([  # RGB图像预处理
        transforms.Resize([128, 416]),
        transforms.ToTensor(),
        #transforms.Pad([78, 0]),  # 上下填充78
        # transforms.CenterCrop(224),
        #transforms.Normalize(mean=[0.377, 0.402, 0.383], std=[0.304, 0.315, 0.319]),
    ])
    image = transform1(image).type(tensor_type)
    return image


def transform_depth(depth, tensor_type='torch.FloatTensor'):
    transform2 = transforms.Compose([  # 深度图像预处理
        #transforms.Resize([128, 416]),
        transforms.ToTensor(),
        #transforms.Pad([78, 0]),
        #transforms.Normalize((1.734,), (8.001,)),  # 这里均值和标准差是通过计算整个训练集图像所得
    ])
    depth = transform2(depth).type(tensor_type)
    return depth


"""
def transform_fwarp(fwarp_image, tensor_type = 'torch.FloatTensor'):
    transforms3 = transforms.Compose([  # RGB图像预处理
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.109, 0.114, 0.111], std=[0.238, 0.245, 0.243]),
    ])
    image = transforms3(fwarp_image).type(tensor_type)
    return fwarp_image"""


def transform_bwarp(bwarp_image, tensor_type = 'torch.FloatTensor'):
    transforms4 = transforms.Compose([  # RGB图像预处理
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.109, 0.114, 0.111], std=[0.238, 0.245, 0.243]),
    ])
    image = transforms4(bwarp_image).type(tensor_type)
    return bwarp_image


def transform_pose(p, tensor_type='torch.FloatTensor'):
    pmean = torch.tensor([79.677, 59.714, 4.719, -0.002, -0.008, 0.352])  # pose训练集的均值
    pstd = torch.tensor([344.457, 243.536, 19.442, 0.032, 0.033, 1.815])  # pose训练集的方差
    return ((p-pmean)/pstd).type(tensor_type)


def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.

    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to

    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform1 = transforms.Compose([  # RGB图像预处理
        transforms.ToTensor(),
        transforms.Resize([224, 68]),
        transforms.Pad([78, 0]),  # 上下填充78
        #transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.109, 0.114, 0.111], std=[0.238, 0.245, 0.243]),
    ])

    transform2 = transforms.Compose([  # 深度图像预处理
        # transforms.ToPILImage(),  # 这里又会自动转变为RGB图像
        # transforms.Grayscale(1),  转换成灰度图像
        transforms.ToTensor(),
        transforms.Resize([224, 68]),
        transforms.Pad([78, 0]),
        #transforms.CenterCrop(224),
        # 将其先由HWC转置为CHW格式，再转为float后每个像素除以255：和kitti深度图像预处理操作一样，这里有一个问题是只对uint8才除以255，所以这里的操作仅为由HWC转置为CHW格式
        # transfer16_01,
        transforms.Normalize((0.501,), (3.580,)),  # 这里均值和标准差是通过计算整个训练集图像所得
    ])
    """
    transform = transforms.ToTensor()
    # Convert single items
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth',
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    # Convert lists
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]"""

    # Convert single items
    for key in filter_dict(sample, ['rgb', 'rgb_last', 'rgb_next']):
        sample[key] = transform1(sample[key]).type(tensor_type)
    for key in filter_dict(sample, ['depth', 'depth_last', 'depth_next']):
        sample[key] = transform2(sample[key]).type(tensor_type)
    for key in filter_dict(sample, ['pose', 'pose_last', 'pose_next']):
        sample[key] = transform_pose(sample[key])

    '''# Convert lists
    for key in filter_dict(sample, ['rgb_last', 'rgb_next']):
        sample[key] = [transform1(k).type(tensor_type) for k in sample[key]]
    for key in filter_dict(sample, ['depth_last', 'depth_next']):
        sample[key] = [transform2(k).type(tensor_type) for k in sample[key]]
    for key in filter_dict(sample, ['pose_last', 'pose_next']):
        sample[key] = [transform_pose(k) for k in sample[key]]'''

    # Return converted sample
    return sample

########################################################################################################################


def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.

    Parameters
    ----------
    sample : dict
        Input sample

    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """
    # Duplicate single items
    for key in filter_dict(sample, [
        'rgb'
    ]):
        sample['{}_original'.format(key)] = sample[key].copy()
    # Duplicate lists
    for key in filter_dict(sample, [
        'rgb_context'
    ]):
        sample['{}_original'.format(key)] = [k.copy() for k in sample[key]]
    # Return duplicated sample
    return sample

########################################################################################################################
'''
def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.
    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to
    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """
    transform = transforms.ToTensor()
    # Convert single items
    for key in filter_dict(sample, [
        'rgb', 'rgb_original', 'depth',
    ]):
        sample[key] = transform(sample[key]).type(tensor_type)
    # Convert lists
    for key in filter_dict(sample, [
        'rgb_context', 'rgb_context_original', 'depth_context'
    ]):
        sample[key] = [transform(k).type(tensor_type) for k in sample[key]]
    # Return converted sample
    return sample'''
########################################################################################################################


def train_transforms(sample, image_shape):
    """
    Training data augmentation transformations

    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape

    Returns
    -------
    sample : dict
        Augmented sample
    """

    #if len(image_shape) > 0:
        #sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    sample = to_tensor_sample(sample)
    return sample


def to_tensor_transforms(sample):
    sample = to_tensor_sample(sample)
    return sample
