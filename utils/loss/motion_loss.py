#《《Unsupervised Monocular Depth Learning in Dynamic Scenes》》
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.loss import consistency_loss, regularizers, transform_depth_map

from dataset.utils.warper import invert_intrinsics_matrix

# T(u,v) = T(motion)decoder求解 + T(ego):这个已知
# validity mask可以改成Motion segmentation


def _get_intrinsics_mat_pyramid(intrinsics_mat, num_scales):
    """Returns multiple intrinsic matrices for different scales.

    Args:
        intrinsics_mat: <float32>[B, 3, 3] tensor containing the intrinsics matrix
        at the original scale.
        num_scales: integer indicating *total* number of matrices to return.  If
        `num_scales` is 1, the function just returns the input matrix in a list.

    Returns:
        List containing `num_scales` intrinsics matrices, each with shape
        <float32>[B, 3, 3].  The first element in the list is the input
        intrinsics matrix and the last element is the intrinsics matrix for the
        coarsest scale.
    """
    # intrinsics_mat: [B, 3, 3]
    intrinsics_mat_pyramid = [intrinsics_mat]
    # Scale the intrinsics accordingly for each scale.
    for s in range(1, num_scales):
        fx = intrinsics_mat[:, 0, 0] / 2 ** s
        fy = intrinsics_mat[:, 1, 1] / 2 ** s
        cx = intrinsics_mat[:, 0, 2] / 2 ** s
        cy = intrinsics_mat[:, 1, 2] / 2 ** s
        intrinsics_mat_pyramid.append(_make_intrinsics_matrix(fx, fy, cx, cy))
    return intrinsics_mat_pyramid


def _make_intrinsics_matrix(fx, fy, cx, cy):
    """Constructs a batch of intrinsics matrices given arguments..

    Args:
        fx: <float32>[B] tensor containing horizontal focal length.
        fy: <float32>[B] tensor containing vertical focal length.
        cx: <float32>[B] tensor containing horizontal principal offset.
        cy: <float32>[B] tensor containing vertical principal offset.

    Returns:
        <float32>[B, 3, 3] tensor containing batch of intrinsics matrices.
    """
    # fx, fy, cx, cy: [B]
    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)
    r1 = torch.stack([fx, zeros, cx], dim=-1)
    r2 = torch.stack([zeros, fy, cy], dim=-1)
    r3 = torch.stack([zeros, zeros, ones], dim=-1)
    intrinsics = torch.stack([r1, r2, r3], dim=1)
    return intrinsics


def _min_pool2d(input_, ksize, strides, padding):
    return -torch.nn.MaxPool2d(ksize, strides, padding=None)(-input_)


def _get_pyramid(img, num_scales, pooling_fn=torch.nn.AvgPool2d):
    """Generates a pyramid from the input image/tensor at different scales.

    This function behaves similarly to `tfg.image.pyramid.split()`.  Instead of
    using an image resize operation, it uses average pooling to give each
    input pixel equal weight in constructing coarser scales.

    Args:
        img: [B, height, width, C] tensor, where B stands for batch size and C
        stands for number of channels.
        num_scales: integer indicating *total* number of scales to return.  If
        `num_scales` is 1, the function just returns the input image in a list.
        pooling_fn: A callable with tf.nn.avg_pool2d's signature, to be used for
        pooling `img` across scales.

    Returns:
        List containing `num_scales` tensors with shapes
        [B, height / 2^s, width / 2^s, C] where s is in [0, num_scales - 1].  The
        first element in the list is the input image and the last element is the
        resized input corresponding to the coarsest scale.
    """
    pyramid = [img]
    for _ in range(1, num_scales):
        # Scale image stack.
        last_img = pyramid[-1]
        scaled_img = pooling_fn(2, 2, padding=None)(last_img)
        pyramid.append(scaled_img)
    return pyramid


# 一共三个loss
class MotionLoss(nn.Module):

    def __init__(self):
        super(MotionLoss, self).__init__()
        self.default_weights = {
                            'rgb_consistency': 1.0,
                            'ssim': 3.0,
                            #'depth_consistency': 0.05,
                            #'depth_smoothing': 0.05,
                            #'rotation_cycle_consistency': 1e-3,
                            #'translation_cycle_consistency': 5e-2,
                            #'depth_variance': 0.0,
                            'motion_smoothing': 1.0,
                            'motion_drift': 0.2,
                        }  # 每个loss的权重
        self.default_params = {
            #'target_depth_stop_gradient': True,
            'scale_normalization': False,
            'num_scales': 1,
        }
        self._output_endpoints = {}

    # 初始化：每个loss都初始化为当前的权重（即loss值为1）
    def _reinitialise_losses(self, device):
        _losses = {k: torch.tensor(0.0).to(device) for k in self.default_weights.keys()}
        return _losses

    def forward(self, Ir, It, Dr, Dt, pose, intrinsics_, residual_translation, target_mask, source_mask, seg_loss=None):
        #Ir = endpoints['Ir']  # Bx3xHxW:Tensor[It...It+batch-1, It+1.....It+batch]
        #It = endpoints['It']  # Bx3xHxW:Tensor[It+1.....It+batch]

        residual_translation = residual_translation.transpose(1, 2).transpose(2, 3)

        rotation = pose[:, 3:].view(-1, 3)   # FIXME 源代码使用的欧拉角，这里输入进来的是轴角，还需转换
        background_translation = pose[:, :3].view(-1, 3)
        #residual_translation = residual_translation  # [B, 3, H, W]
        intrinsics_mat = intrinsics_  # [B, 3, 3]
        _losses = self._reinitialise_losses(It.device)  # 同属与一个device

        # Weight applied to all losses at this scale: 0.5
        #scale_w = 1.0 / 2 ** 1

        # 训练模型时可能出现预测深度的global scale(不重要)导致训练不稳定的情况。为了解决这个问题，对预测深度进行了归一化处理
        mean_depth = torch.mean(torch.cat((Dr, Dt), dim=0))  # 这里的depth是需要target还是source的 TODO
        if self.default_params['scale_normalization']:
            Dr /= mean_depth
            Dt /= mean_depth
        # 根据论《wild》 离camera越近惩罚越重
            background_translation /= mean_depth  # ？
            residual_translation /= mean_depth

        # T(u, v) = T(ego) + T(motion)：这里的background_translation原形状是(B,3,1,1)现在多加一个维度，相当于motion map每个像素上加
        translation = torch.add(residual_translation, background_translation.view(-1, 1, 1, 3))  # TODO:转换成mask


        #反方向的总平移=反方向的背景平移加反方向的运动平移
        #flipped_translation = (flipped_residual_translation + flipped_background_translation.view(-1, 1, 1, 3))

        intrinsics_mat_inv = invert_intrinsics_matrix(intrinsics_mat)

        # 加上物体的运动，对深度图重新行投影，最后得到五个参数，其中pixel_x和pixel_y分别表示原像素的偏移值，depth表示新投影得到的深度图
        # 这里的transform就是将Dr投影到了Dt坐标系，基于这个也可以直接转换rgb图像
        transformed_depth = transform_depth_map.using_motion_vector(  # batchsize对不上
            torch.squeeze(Dr, dim=1), translation, rotation, 'euler_angle',  # translation.shape(b,h,w,3),rotation.shape(b,1,3)  TODO 还有一个问题是1,3的旋转矩阵怎么和有h,w的trans结合
            intrinsics_mat, intrinsics_mat_inv)  # Dt'

        # 计算有效区域即没有被遮挡住的区域
        geo_valid = transformed_depth.mask.unsqueeze(1).to(torch.uint8)  # 这里的mask指的是投影后的坐标正确落在图像区域内的位置

        warped_seg = F.grid_sample(source_mask.to(dtype=torch.float32), transformed_depth.pixel_xy)
        vis_valid = target_mask.eq(warped_seg).to(torch.uint8)  # 返回的是未被遮挡即有效的区域
        occ_error = torch.max(geo_valid-vis_valid, torch.tensor(0).to(device=It.device))  # vis_mask存在着不准确的运动估计引起的偏移和遮挡，因此遮挡区域多有效区域少，这一区域也应该计算进loss
        # 再圈出有效区域内的运动物体，保证计算出的运动物体的偏移准确度
        validity_mask = occ_error + geo_valid

        loss_endpoints = (  # tuple数据
            consistency_loss.rgbd_and_motion_consistency_loss(
                transformed_depth,
                Ir,
                Dt,
                It,
                validity_mask))

        # 根据相机平移进行归一化
        normalized_trans = regularizers.normalize_motion_map(residual_translation, translation)
        # Lreg,mot
        # group smoothness loss→Regularization loss
        ##########################################乘以foreground_mask####################################################
        #weight = seg_loss
        # TODO:求出每个object和背景中最合适的motion map，然后惩罚
        _losses['motion_smoothing'] = regularizers.l1smoothness(
            normalized_trans, self.default_weights['motion_drift'] == 0)
        # sparsity loss
        _losses['motion_drift'] = regularizers.sqrt_sparsity(
            normalized_trans)

        # Lrgb: photometric consistency
        _losses['rgb_consistency'] = loss_endpoints['rgb_error']
        _losses['ssim'] = 0.5 * loss_endpoints['ssim_error']

        # Llabel:label consistency TODO 之后可以加上分割的consistency los

        #Lcyc中第二个公式:删掉的原因是逆的物体运动估计需要将原顺序调换，最后对应的真实标签也需要两张对应的：这里还不确定是不是要改成这样
        #_losses['translation_cycle_consistency'] = scale_w * loss_endpoints['translation_error']

        self._output_endpoints['depth_proximity_weight'] = loss_endpoints['depth_proximity_weight']
        self._output_endpoints['trans'] = translation

        for k, w in self.default_weights.items():
            # multiply by 2 to match the scale of the old code.
            # 总loss乘以对应的权重
            _losses[k] = _losses[k] * w
        losses = sum(_losses.values())
        return losses


if __name__ == '__main__':
    source_view = torch.randn(4, 3, 128, 416).cuda()
    target_view = torch.randn(4, 3, 128, 416).cuda()
    source_depth = torch.randn(4, 1, 128, 416).cuda()
    target_depth = torch.randn(4, 1, 128, 416).cuda()
    axisangle = torch.randn(4, 1, 3).cuda()
    translation = torch.randn(4, 1, 3).cuda()
    intrinsics_ = torch.randn(4, 3, 3).cuda()

    sample = {}
    sample.update({
        # 'rgb': image,
        # 'depth': depth,
        # 为了重新投影并计算motion loss函数,
        'Ir': source_view,
        'It': target_view,
        'Dr': source_depth,
        'Dt': target_depth,
        'axisangle': axisangle[:, 0].squeeze(0),  # 轴角
        'trans': translation[:, 0].squeeze(0),
        'intrinsics_mat': intrinsics_,
    })
    residual_t = torch.randn(4, 3, 128, 416).cuda()
    target_mask = torch.argmax(torch.randn(4, 9, 128, 416), dim=1, keepdim=True).cuda()

    source_mask = torch.argmax(torch.randn(4, 9, 128, 416), dim=1, keepdim=True).cuda()

    criterion = MotionLoss()
    loss = criterion(source_view, target_view, source_depth, target_depth, axisangle[:, 0].squeeze(0),
                     translation[:, 0].squeeze(0), intrinsics_, residual_t, target_mask, source_mask)
    print(loss)