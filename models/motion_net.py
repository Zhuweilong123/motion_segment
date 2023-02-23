from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn.init import xavier_uniform_, zeros_
#from pose_net import PosePredictionNet
#from deeplabv3_plus import SharedEncoder


# 来自《wild》
# 《Attentive and Contrastive Learning for Joint Depth and Motion Field Estimation》整体的思路：
# 输入image concat predicted depth[b,8,h,w]→升维→[b,1024,h,w]→降维→[b,6,1,1]相机自运动→降维→[b,3,h,w]→refine


def create_scales(constraint_minimum):
    """Creates variables representing rotation and translation scaling factors.
      Args:
        constraint_minimum: A scalar, the variables will be constrained to not fall
          below it.
      Returns:
        Two scalar variables, rotation and translation scale.
      """
    initialise = 0.01

    def constraint(x):
        return float(
            nn.ReLU(inplace=False)(torch.tensor(x - constraint_minimum).type(torch.float))) + constraint_minimum

    rot_scale = constraint(initialise)
    trans_scale = constraint(initialise)
    return rot_scale, trans_scale


"""     
    rot_scale, trans_scale = create_scales(0.001)
    background_translation *= trans_scale
    residual_translation *= trans_scale
    rotation *= rot_scale
    if self.auto_mask:  # auto_mask: True to automatically masking out the residual translation by thresholding on their mean values.
        residual_translation = self._mask(residual_translation)
    image_height, image_width = x.shape[2], x.shape[3]

    return (rotation, background_translation.reshape(-1, 3), residual_translation.clone().reshape(-1, 128, 416, 3),
            intrinsic_mat)
"""


# 可以理解为motion field任务的decoder block，每个block输出的都是3维，只是尺寸不一样
class RefinementLayer(nn.Module):
    def __init__(self, num_channel, dims, num_motion_fields=3):
        """Refines a motion field using features from another layer:优化运动场
          This function builds an element of a UNet-like architecture. `motion_field`
          has a lower spatial resolution than `layer`. First motion_field is resized to
          `layer`'s spatial resolution using bilinear interpolation, then convolutional
          filters are applied on `layer` and the result is added to the upscaled
          `motion_field`.
          Args:
              num_channel: 每一层输入的通道数量
              dims: encoder倒序输出的每一层的特征图大小
        """
        super(RefinementLayer, self).__init__()
        self.num_channel = num_channel  # 与encoder连接的通道数
        self.num_mid_channel = max(4, self.num_channel)  # 中间通道
        self.dims = dims
        self.num_motion_fields = num_motion_fields  # 输出三维
        # same padding by hard coded for now
        self.conv1 = nn.Conv2d(self.num_motion_fields + self.num_channel,  # 3+通道数
                               self.num_mid_channel, 3, padding=1)
        self.conv2_1 = nn.Conv2d(self.num_motion_fields + self.num_channel,
                                 self.num_mid_channel, 3, padding=1)
        self.conv2_2 = nn.Conv2d(self.num_mid_channel, self.num_mid_channel, 3,
                                 padding=1)
        self.conv3 = nn.Conv2d(self.num_mid_channel * 2,
                               self.num_motion_fields, 1,
                               bias=False)  # 1x1的卷积核，目的是将中间通道转换成3维
        self.relu = nn.ReLU(inplace=True)

    def forward(self, motion_field, feature):
        """
        Args:
            motion_field: a Tensor of shape [B, m, h1, w1]. m is the number of
            dimensions in the motion field, for example, 3 in case of a 3D translation
            field.
            feature: Tensor of shape [B, h2, w2, c].
        Returns:
            A Tensor of shape [B, h2, w2, m], obtained by upscaling motion_field to
            h2, w2"""
        # Pytorch does not support the half_pixel_center argument
        # it induces difference in the network output compared
        # to the official tensorflow repo
        upsampled_motion_field = nn.functional.interpolate(
            motion_field, self.dims, mode='bilinear')
        x = torch.cat((upsampled_motion_field, feature), dim=1)
        output1 = self.relu(self.conv1(x))  # 拼接后的通道数转成feature特征数
        output2 = self.relu(self.conv2_1(x))  # 和conv1层相同
        output2 = self.relu(self.conv2_2(output2))  # 这里是先将拼接后的通道数转成feature特征数，再进行一次卷积操作，这次通道数不变
        output = torch.cat((output1, output2), dim=1)  # 得到的通道数为num_channel*2
        output = upsampled_motion_field + self.conv3(output)  # identity加上通过1x1卷积转换到原通道数的特征图
        return output


class MotionNet(nn.Module):
    def __init__(self, input_dims, bottleneck_dims, auto_mask=False):
        super(MotionNet, self).__init__()
        #self.backbone = SharedEncoder()

        self.input_dims = input_dims  # [h,w],要求h和w都是32的倍数,kitti中一般默认是[128, 416]
        #self.num_input_images = num_input_images  # 单次输入到网络中的图像数量，一般为2
        self.bottleneck_dims = bottleneck_dims  # 编码器里中每一层输出的特征图大小（从后到前）
        self._init_motion_field_net()
        self.auto_mask = auto_mask

    def _init_motion_field_net(self):
        self.num_ch_bottleneck = [32, 16, 24, 32, 64, 96, 160,
                                  320]  # [16, 32, 64, 128, 256, 512, 1024] (原resnet和现Mobilenetv2对比)
        self.refinement1 = self._make_refinement_layer(
            self.num_ch_bottleneck[-1],  # 1024→320
            self.bottleneck_dims[-1])  # 该层对应的分辨率[1,4]→[16,52]
        self.refinement2 = self._make_refinement_layer(
            self.num_ch_bottleneck[-2],  # 512→160
            self.bottleneck_dims[-2])  # [2,7]→[16,52]
        self.refinement3 = self._make_refinement_layer(
            self.num_ch_bottleneck[-3],  # 256→96
            self.bottleneck_dims[-3])  # [4,13]→[16,52]
        self.refinement4 = self._make_refinement_layer(
            self.num_ch_bottleneck[-4],  # 128→64
            self.bottleneck_dims[-4])  # [8,26]→[16,52]
        self.refinement5 = self._make_refinement_layer(
            self.num_ch_bottleneck[-5],  # 64→32
            self.bottleneck_dims[-5])  # [16,52]→[16,52]
        self.refinement6 = self._make_refinement_layer(
            self.num_ch_bottleneck[-6],  # 32→24
            self.bottleneck_dims[-6])  # [32,104]→[32,104]
        self.refinement7 = self._make_refinement_layer(
            self.num_ch_bottleneck[-7],  # 16→16
            self.bottleneck_dims[-7])  # [64,208]→[64,208]
        # 得到最终的三维motion field
        self.refinement8 = self._make_refinement_layer(  # 每个refine_layer输出的都是3维的特征
            self.num_ch_bottleneck[-8],  # 6→32
            self.bottleneck_dims[-8])  # [128,416]→[64,208],还需要放大一倍
        # self.final_conv = nn.Conv2d()

    def _make_refinement_layer(self, ch, dims):
        return RefinementLayer(ch, dims)

    # auto_mask: True to automatically masking out the residual translation by thresholding on their mean values.
    def _mask(self, x):
        sq_x = torch.sqrt(torch.sum(x ** 2,
                                    dim=1, keepdim=True))
        mean_sq_x = torch.mean(sq_x, dim=(0, 2, 3))
        mask_x = (sq_x > mean_sq_x).type(x.dtype)
        x = x * mask_x
        return x

    def forward(self, x):  # translation是求到的相机平移向量这里可以改成mobilenet最终或者某层的输出，一定要满足维度为3，[b,3,1,1]，因为mobilenetv2输出的最后的通道数是320不是3，所以这里需要再加上一个卷积层转换通道数
        # features是list,每个元素分别表示编码器返回的七个卷积层的输出(tensor)
        # residual_translation = self.conv(translation, )
        '''residual_translation = self.refinement1(translation, features[7])  #
        residual_translation = self.refinement2(
            residual_translation, features[6])
        residual_translation = self.refinement3(
            residual_translation, features[5])
        residual_translation = self.refinement4(
            residual_translation, features[4])
        residual_translation = self.refinement5(
            residual_translation, features[3])
        residual_translation = self.refinement6(
            residual_translation, features[2])
        residual_translation = self.refinement7(
            residual_translation, features[1])
        #print(residual_translation.shape)
        residual_translation = self.final_refinement(
            residual_translation, features[0]
        )'''
        features, translation = x[:-1], x[-1]
        residual_translation = self.refinement1(translation, features[7])  #
        residual_translation = self.refinement2(
            residual_translation, features[6])
        residual_translation = self.refinement3(
            residual_translation, features[5])
        residual_translation = self.refinement4(
            residual_translation, features[4])
        residual_translation = self.refinement5(
            residual_translation, features[3])
        residual_translation = self.refinement6(
            residual_translation, features[2])
        residual_translation = self.refinement7(
            residual_translation, features[1])
        # print(residual_translation.shape)
        residual_translation = self.refinement8(
            residual_translation, features[0]
        )
        # residual_translation = self.final_refinement(residual_translation, features[0])
        # residual_translation = self.final_conv(residual_translation)
        # 还要再进行一次上采样
        residual_translation = nn.functional.interpolate(
            residual_translation, scale_factor=2, mode='bilinear')
        if self.auto_mask:  # auto_mask: True to automatically masking out the residual translation by thresholding on their mean values.
            residual_translation = self._mask(residual_translation)
        return residual_translation


if __name__ == "__main__":
    x = torch.randn((1, 6, 128, 416))  # It,It^拼接
    x = x.to(device='cuda')

    mn = MotionNet(input_dims=(128, 416),
                   bottleneck_dims=[(64, 208), (64, 208), (32, 104), (16, 52), (16, 52), (16, 52), (16, 52), (16, 52)],
                   auto_mask=True).to(device='cuda')  # 这里的 auto-mask可以替换成语义分割任务的结果
    o = mn(x)
