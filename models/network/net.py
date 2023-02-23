import torch
from torch import nn
import math
import torch.utils.model_zoo as model_zoo

import utils
from ..backbones.ResNet import Bottleneck_coordatt
from ..attention.CBAM import MultiModalAttention
from ..network.decoder import TransBasicBlock
from torch.utils.checkpoint import checkpoint

# 一些RGBD语义分割的最新论文会在深度特征和rgb特征融合时结合一些attention方法，即不是通过1：:1的方法直接相加

class Rigidnet(nn.Module):
    # num_class也是最后一层输出的通道数
    def __init__(self, num_classes=3, pretrained=False):
        super(Rigidnet, self).__init__()
        # 两处需要融合的位置：深度特征+视觉特征，下采样特征+上采样特征 #

        """ shared encoder """
        block = Bottleneck_coordatt  # 添加了coordinate attention机制的基础模块
        layers = [3, 4, 6, 3]  # 基于resnet-50
        # original resnet
        """ Encoder for warped image diff """
        self.inplanes = 64
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        """ Encoder for depth channel """
        self.inplanes = 64
        # 两个深度图拼接
        self.conv1_d = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1_d = nn.BatchNorm2d(64)
        self.layer1_d = self._make_layer(block, 64, layers[0])
        self.layer2_d = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_d = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_d = self._make_layer(block, 512, layers[3], stride=2)

        """cat + channel attention"""
        self.mm0 = MultiModalAttention(64)
        self.mm1 = MultiModalAttention(64 * 4)
        self.mm2 = MultiModalAttention(128 * 4)
        self.mm3 = MultiModalAttention(256 * 4)
        self.mm4 = MultiModalAttention(512 * 4)

        """ Decoder for RigidSeg"""
        transblock = TransBasicBlock

        self.inplanes = 512
        self.deconv1_s = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2_s = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3_s = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4_s = self._make_transpose(transblock, 64, 3, stride=2)

        # final block
        self.inplanes = 64
        self.final_conv_s = self._make_transpose(transblock, 64, 3)

        self.final_deconv_s = nn.ConvTranspose2d(self.inplanes, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv_s = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv_s = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv_s = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv_s = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)

        """ Decoder for MotionField"""
        transblock = TransBasicBlock

        self.inplanes = 512
        self.deconv1_m = self._make_transpose(transblock, 256, 6, stride=2)
        self.deconv2_m = self._make_transpose(transblock, 128, 4, stride=2)
        self.deconv3_m = self._make_transpose(transblock, 64, 3, stride=2)
        self.deconv4_m = self._make_transpose(transblock, 64, 3, stride=2)

        self.agent0 = self._make_agent_layer(64, 64)
        self.agent1 = self._make_agent_layer(64 * 4, 64)
        self.agent2 = self._make_agent_layer(128 * 4, 128)
        self.agent3 = self._make_agent_layer(256 * 4, 256)
        self.agent4 = self._make_agent_layer(512 * 4, 512)

        # final block
        self.inplanes = 64
        self.final_conv_m = self._make_transpose(transblock, 64, 3)

        self.final_deconv_m = nn.ConvTranspose2d(self.inplanes, num_classes, kernel_size=2,
                                               stride=2, padding=0, bias=True)

        self.out5_conv_m = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=True)
        self.out4_conv_m = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, bias=True)
        self.out3_conv_m = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)
        self.out2_conv_m = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if pretrained:
            self._load_resnet_pretrained()

    def _make_transpose(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,  # self.inplanes 默认为调用前的最新更新
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)

    # 编码器输出的通道数非常多，可以直接将低channel size的特征图投影到解码器块中，降低内存消耗
    # 编码器low-level的特征图怎么融合解码器high-level的特征图
    def _make_agent_layer(self, inplanes, planes):

        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def _load_resnet_pretrained(self):
        pretrain_dict = model_zoo.load_url(utils.model_urls['resnet50'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if k.startswith('conv1'):  # the first conv_op
                    model_dict[k] = v
                    model_dict[k.replace('conv1', 'conv1_d')] = torch.mean(v, 1).data. \
                        view_as(state_dict[k.replace('conv1', 'conv1_d')])

                elif k.startswith('bn1'):
                    model_dict[k] = v
                    model_dict[k.replace('bn1', 'bn1_d')] = v
                elif k.startswith('layer'):
                    model_dict[k] = v
                    model_dict[k[:6] + '_d' + k[6:]] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    # encoder
    def forward_downsample(self, res_warp, depth):

        x = self.conv1(res_warp)  # res_warp 是两个方向求到的warped image的差值，shape为[B, 6, H, W]
        x = self.bn1(x)
        x = self.relu(x)
        depth = self.conv1_d(depth)
        depth = self.bn1_d(depth)
        depth = self.relu(depth)

        #fuse0 = x + depth
        # concate之后通道数double了
        fuse0 = self.mm0(x, depth)

        x = self.maxpool(fuse0)
        depth = self.maxpool(depth)

        # block 1
        x = self.layer1(x)
        depth = self.layer1_d(depth)
        #fuse1 = x + depth
        fuse1 = self.mm1(x, depth)

        # block 2
        x = self.layer2(fuse1)
        depth = self.layer2_d(depth)
        #fuse2 = x + depth
        fuse2 = self.mm2(x, depth)

        # block 3
        x = self.layer3(fuse2)
        depth = self.layer3_d(depth)
        #fuse3 = x + depth
        fuse3 = self.mm3(x, depth)

        # block 4
        x = self.layer4(fuse3)
        depth = self.layer4_d(depth)
        #fuse4 = x + depth
        fuse4 = self.mm4(x, depth)

        return fuse0, fuse1, fuse2, fuse3, fuse4

    # decoder for seg
    def seg_upsample(self, fuse0, fuse1, fuse2, fuse3, fuse4):

        # agent类似resnet里面的恒等映射块，
        agent4 = self.agent4(fuse4)
        # upsample 1
        x = self.deconv1_s(agent4)
        if self.training:
            out5 = self.out5_conv_s(x)
        x = x + self.agent3(fuse3)
        # upsample 2
        x = self.deconv2_s(x)
        if self.training:
            out4 = self.out4_conv_s(x)
        x = x + self.agent2(fuse2)
        # upsample 3
        x = self.deconv3_s(x)
        if self.training:
            out3 = self.out3_conv_s(x)
        x = x + self.agent1(fuse1)
        # upsample 4
        x = self.deconv4_s(x)
        if self.training:
            out2 = self.out2_conv_s(x)
        x = x + self.agent0(fuse0)
        # final
        x = self.final_conv_s(x)
        out = self.final_deconv_s(x)

        if self.training:
            return out, out2, out3, out4, out5

        return out

    # decoder for motion
    def motion_upsample(self, fuse0, fuse1, fuse2, fuse3, fuse4):

        # agent类似resnet里面的恒等映射块，
        agent4 = self.agent4(fuse4)
        # upsample 1
        x = self.deconv1_m(agent4)
        if self.training:
            out5 = self.out5_conv_m(x)
        x = x + self.agent3(fuse3)
        # upsample 2
        x = self.deconv2_m(x)
        if self.training:
            out4 = self.out4_conv_m(x)
        x = x + self.agent2(fuse2)
        # upsample 3
        x = self.deconv3_m(x)
        if self.training:
            out3 = self.out3_conv_m(x)
        x = x + self.agent1(fuse1)
        # upsample 4
        x = self.deconv4_m(x)
        if self.training:
            out2 = self.out2_conv_m(x)
        x = x + self.agent0(fuse0)
        # final
        x = self.final_conv_m(x)
        out = self.final_deconv_m(x)

        if self.training:
            #return out, out2, out3, out4, out5
            return out

        return out

    def forward(self, res_warp, depth, phase_checkpoint=False):

        if phase_checkpoint:
            depth.requires_grad_()
            fuses = checkpoint(self.forward_downsample, res_warp, depth)
            out1 = checkpoint(self.seg_upsample, *fuses)
            ou2 = checkpoint(self.motion_upsample, *fuses)
        else:
            # encoder
            fuses = self.forward_downsample(res_warp, depth)
            # decoder
            out1 = self.seg_upsample(*fuses)
            out2 = self.motion_upsample(*fuses)
        return out1, out2

"""
# 查看模型参数数量
# 利用高阶 API 查看模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ca_res50 =Motionfield().to(device)
#print(ca_res50)
x = torch.rand(1, 6, 224, 224)
x = x.type(torch.cuda.FloatTensor)
y = torch.rand(1, 2, 224, 224)
y = y.type(torch.cuda.FloatTensor)
i = ca_res50(x, y)
#print(i.type)
#查看网络的顺序结构，还有网络参数量，网络模型大小
summary(ca_res50, [(6, 224, 224), (2, 224, 224)])"""
