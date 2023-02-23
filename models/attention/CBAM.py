import torch
import torch.nn as nn
from os import listdir
import os


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# 每一块的深度特征和pose特征进行融合时加入attention机制并回归成原来的维度
class MultiModalAttention(nn.Module):
    def __init__(self, inplanes):
        super(MultiModalAttention, self).__init__()
        self.ca = ChannelAttention(inplanes*2)

        self.conv = nn.Sequential(
            nn.Conv2d(inplanes*2, inplanes, 1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True))

    def forward(self, x, y):
        out = torch.cat([x, y], dim=1)  #通道数*2
        out = self.ca(out) * out
        # 再利用1x1的卷积核使得通道数减半
        return self.conv(out)

