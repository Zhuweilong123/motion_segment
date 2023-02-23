import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from __future__ import division
import torch
from torch import nn
import torch.nn.functional as F
from dataset.utils.warper import inverse_warp2, inverse_warp
import math

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 用于训练深度估计和姿态估计网络
# motion mask 改成 motion segmentation或的结果

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


compute_ssim_loss = SSIM().to(device)


# photometric loss
# geometry consistency loss
def compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths, poses, poses_inv, max_scales, with_ssim, with_mask, with_auto_mask, padding_mode):

    photo_loss = 0
    geometry_loss = 0

    num_scales = min(len(tgt_depth), max_scales)  # len(tgt_depth):batch_size
    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, poses, poses_inv):
        for s in range(num_scales):
            # upsample depth
            b, _, h, w = tgt_img.size()
            tgt_img_scaled = tgt_img
            ref_img_scaled = ref_img
            intrinsic_scaled = intrinsics
            # 除了第一个batch，其他batch都需要通过最近邻插值法，上采样/下采样输出(h,w)大小的张量
            if s == 0:
                tgt_depth_scaled = tgt_depth[s]
                ref_depth_scaled = ref_depth[s]
            else:
                # 最近邻插值
                tgt_depth_scaled = F.interpolate(tgt_depth[s], (h, w), mode='nearest')
                ref_depth_scaled = F.interpolate(ref_depth[s], (h, w), mode='nearest')

            photo_loss1, geometry_loss1 = compute_pairwise_loss(tgt_img_scaled, ref_img_scaled, tgt_depth_scaled, ref_depth_scaled, pose,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)
            # 这里计算的是反方向的损失
            photo_loss2, geometry_loss2 = compute_pairwise_loss(ref_img_scaled, tgt_img_scaled, ref_depth_scaled, tgt_depth_scaled, pose_inv,
                                                                intrinsic_scaled, with_ssim, with_mask, with_auto_mask, padding_mode)

            photo_loss += (photo_loss1 + photo_loss2)
            geometry_loss += (geometry_loss1 + geometry_loss2)

    return photo_loss, geometry_loss


def compute_pairwise_loss(tgt_img, ref_img, tgt_depth, ref_depth, pose, intrinsic, with_ssim, with_mask, with_auto_mask, padding_mode):

    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(ref_img, tgt_depth, ref_depth, pose, intrinsic, padding_mode)

    diff_img = (tgt_img - ref_img_warped).abs().clamp(0, 1)  # 把差值限定在(0, 1)范围内

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    if with_auto_mask == True:
        auto_mask = (diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)).float() * valid_mask
        valid_mask = auto_mask

    if with_ssim == True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped)
        diff_img = (0.15 * diff_img + 0.85 * ssim_map)

    if with_mask == True:
        weight_mask = (1 - diff_depth)  # 深度发生变化的
        diff_img = diff_img * weight_mask

    # compute all loss，只计算有效位置（静态区域）上的差值，但是这里valid mask的来源依旧存在疑问
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

    return reconstruction_loss, geometry_consistency_loss


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    if mask.sum() > 10000:
        mean_value = (diff * mask).sum() / mask.sum()
    else:
        mean_value = torch.tensor(0).float().to(device)
    return mean_value


@torch.no_grad()
def compute_errors(gt, pred, dataset):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0, 0, 0, 0, 0, 0
    batch_size, h, w = gt.size()

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if dataset == 'kitti':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1, x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 80

    if dataset == 'nyu':
        crop_mask = gt[0] != gt[0]
        y1, y2 = int(0.09375 * gt.size(1)), int(0.98125 * gt.size(1))
        x1, x2 = int(0.0640625 * gt.size(2)), int(0.9390625 * gt.size(2))
        crop_mask[y1:y2, x1:x2] = 1
        max_depth = 10

    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0.1) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, max_depth)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


# 计算自适应的marginloss
class AdaptiveMarginLoss(nn.Module):
    def __init__(self, batch, lamda, gama):  # nu:regularization parameter for beta
        super(AdaptiveMarginLoss, self).__init__()
        self.batch = batch  # 当前batch的数量

        self.lamda = lamda  # loss2的权重
        self.gama = gama  # loss3的权重

    def forward(self, Dap, Dan, Dsn, Ddn):

        s_m = torch.ones([1, self.batch])  # 两个margin的初始都是1
        b_m = torch.ones([1, self.batch])
        # 传统Tripletloss的计算公式
        class_loss = torch.clamp(Dap - Dan + self.margin, min=0.0)
        geo_loss = torch.clamp(Dsn - Ddn + self.Mmargin, min=0.0)

        margin_loss = torch.mean(class_loss) + self.lamda * torch.mean(geo_loss)

        #
        adaptive_class = -1 * self.gama * (torch.mean())


        loss = margin_loss  # + adaptive_margin

        return loss









