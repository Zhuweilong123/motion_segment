import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_
from torchsummary import summary

from models.mbv2_ca import mobilenetv2_ca
from models.deeplabv3_plus import DeepLab
from models.motion_net import MotionNet


# multi(cross)-modal fusion module:先利用一个简单的注意力模型
class ModalFusion(nn.Module):
    def __init__(self, in_c, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ModalFusion, self).__init__()
        out_c = in_c//2
        self.merge = nn.Sequential(
            nn.Conv2d(in_c, out_c//reduction, kernel_size=1, bias=True),
            # depth-wise 深度可分离卷积
            nn.Conv2d(out_c//reduction, out_c//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_c//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c//reduction, out_c, kernel_size=1, bias=True),
            norm_layer(out_c)
            )
        self.residual = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, y, task, method="concat"):
        if method == "sum":  # 不增加额外的计算量，但是方法简单，每种模态以1:1的方式占最后输出的
            out = x+y
        if method == "concat":
            out = torch.cat((x, y), dim=1)  # 通道数量x2
            out = self.merge(out)
            # 没有直接利用残差卷积把拼接后的两个特征合转换到一个维度，而是按照不同的任务直接添加对应的feature
            if task == "seg":
                out = x + out
            elif task == "motion":
                out = y + out
            return out

# shared encoder:得到每一层的输出，分别输入到分割解码器和运动场解码器中
class SharedEncoder(nn.Module):
    def __init__(self, backbone, downsample_factor=8, pretrained=True):
        super(SharedEncoder, self).__init__()
        from functools import partial

        model = backbone  # 使用预训练模型
        # self.features   = model.features[:-1]
        self.features = model.features  # 这里model.feature就相当于每一层的网络，最后一层是7x7x320，不包括原mobilenetv2的最后一个卷积层
        # Mobilenetv2的最后一层
        self.conv = model.conv
        self.avgpool = model.avgpool
        self.classifier = model.classifier

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]  # 每个下采样层所处在的位置

        # downsample_factor是下采样次数，这里是根据下采样次数改变mobilenetv2里的stride
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):  # 倒数两个下采样的位置stride改为1，即7和14层
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):  # 最后一个下采样的位置stride改为1，即14层
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        ############输入到deeplabV3plus的解码器中
        # low_level_features = self.features[:4](x)   # 浅层特征：输出的特征通道数为24, 第三块（加上conv），代表的是第0到第3层一共4层网络
        # high_level_features = self.features[4:](low_level_features)  # 深层特征：输出的特征通道数为320，代表的第4层到第17层一共14层网络

        ############输入到motion field的解码器中：首先要确定哪些层特征需要fuse，输入到decoder哪些层中
        features = [0] * 8
        features[0] = self.features[0](x)
        features[1] = self.features[1](features[0])
        features[2] = self.features[2:4](features[1])  # 输入到deeplabv3+里的浅层特征
        features[3] = self.features[4:7](features[2])
        features[4] = self.features[7:11](features[3])
        features[5] = self.features[11:14](features[4])
        features[6] = self.features[14:17](features[5])
        features[7] = self.features[17](features[6])  # 输入到deeplabv3+里的深层特征
        # 最后一个卷积层转换成3维特征
        translation = self.conv(features[7])
        translation = self.avgpool(translation)
        translation = translation.view(translation.size(0), -1)
        translation = self.classifier(translation).unsqueeze(-1).unsqueeze(-1)

        return features, translation


# 将module中所有的conv2d转换成空洞卷积
class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels,
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module


# segmentation +(multi-modal fusion) residual motion field
class RMS(nn.Module):
    def __init__(self, image_size, num_classes, downsample_factor=8):
        super(RMS, self).__init__()
        self.backbone = mobilenetv2_ca()
        self.encoder = SharedEncoder(self.backbone, downsample_factor)  # shared encoder
        self.seg_decoder = DeepLab(num_classes, downsample_factor)
        self.motion_decoder = MotionNet(image_size,
                   bottleneck_dims=[(64, 208), (64, 208), (32, 104), (16, 52), (16, 52), (16, 52), (16, 52), (16, 52)],
                   auto_mask=True)
        in_channel = self.motion_decoder.num_ch_bottleneck
        self.MF = nn.ModuleList()  # 没有放到cuda上  #只使用list包装一些卷积层的话，模型加载到GPU时会被忽略，需要把数据类型改成nn.modulelist
        # multi-modal fusion: 暂时没想出来，写完motion decoder再写
        for i in range(8):
            self.MF.append(ModalFusion(in_channel[i]*2))  # 运动特征和语义特征拼接后通道数扩大了一倍
        self.MF.append(ModalFusion(3*2))

    def forward(self, warp, depth):
        f1, trans1 = self.encoder(warp)  # visual
        f2, trans2 = self.encoder(depth)  # geometric
        fv = []
        fg = []
        # 融合
        for i in range(8):
            if i == 2 or i == 7:  # deeplab只需要其中两层的特征
                fv.append(
                    self.MF[i](f1[i], f2[i], "seg")
                )
            fg.append(
                self.MF[i](f1[i], f2[i], "motion")
            )
        # encoder最后一层特征（3通道）
        fg.append(
            self.MF[8](trans1, trans2, "motion")
        )
        seg1, seg2 = self.seg_decoder(fv)
        residual_trans = self.motion_decoder(fg)
        return seg1, seg2, residual_trans

if __name__ == '__main__':
    x = torch.randn(4, 6, 128, 416)
    x = x.cuda()

    y = torch.randn(4, 6, 128, 416)
    y = y.cuda()
    model = RMS(image_size=(128, 416), num_classes=18)
    model = model.cuda()
    #print(next(model.parameters()).is_cuda)  # 判断模型是否放在了cuda上

    output1, output2, motion = model(x, y)
    #summary(model, input_size=[(6, 128, 416), (6, 128, 416)])  # 该库在内部对输入增加了一个batch维度所以只需要(C，H，W)的tensor
    # 40MB
    print(output1)