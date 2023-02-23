import math
import torch
import torch.nn as nn
from models.attention import ChannelAttention, SpatialAttention
from models.backbones.ResNet import Bottleneck, Bottleneck_am

#原做法是一种late fuse：现在改成middle fuse

# 提取深度特征512*7*7
class DepthNet(nn.Module):
    def __init__(self, block, layers, num_classes=12):
        self.inplanes = 64
        super(DepthNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 最后一个卷积层特征512*7*7，即现在的视觉特征空间

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  # 先升维再降采样
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 每一层一共两个blocks
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x是深度

        x = self.conv1(x)  # batchsize降为了64，这里不太理解为什么batchsize会发生变化
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x  # 相似度在0~1范围内


# 提取姿态特征512*7*7→把pose、depth和rgb融合成warped image，输入特征shape变成了b*3*H*W
class PoseNet(nn.Module):
    def __init__(self, block, layers, dim_p=12, bottom_height=7, bottom_width=7):
        self.inplanes = 64
        self.bottom_height = bottom_height
        self.bottom_width = bottom_width
        super(PoseNet, self).__init__()
        self.l1 = nn.Linear(dim_p, 64 * bottom_width * bottom_height)  # 线性转换
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)  # 最后一个卷积层特征512*7*7，即现在的视觉特征空间
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(  # 先升维再降采样
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 每一层一共两个blocks
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x是pose
        # print(x.shape)  # torch.Size([64, 2, 6])
        h = self.l1(x).view(-1, 64, self.bottom_height, self.bottom_width)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        return h


# 计算两个特征图的余弦相似度并归一化，这里的问题是最终输出的相似度的维度应该设定为多少？
class CrossModalAttention(nn.Module):
    def __init__(self, dim=1024):  # __init__又给打成了__int__
        super(CrossModalAttention, self).__init__()
        #self.pres = PoseNet(BasicBlock1, [2, 2, 2, 2])  # 经过了空间注意力机制的计算，留下来的是更重要的区域，这里有一个问题，深度和姿态真的需要attention机制吗，意义大吗？
        self.pres = PoseNet(Bottleneck, [3, 4, 6, 3])
        #self.dres = DepthNet(BasicBlock2, [2, 2, 2, 2])  # 经过了时间注意力机制的计算，留下来的是关注某通道能量较强的像素点
        self.dres = DepthNet(Bottleneck_am, [3, 4, 6, 3])  #对深度图进行了注意力加强

        self.relu = nn.ReLU(inplace=True)  # 删除小于0的权重
        self.sigmoid = nn.Sigmoid()  # 权重归一化
        self.avgp = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(1024, 1)  # pose和depth拼接后
        self.fc2 = nn.Linear(512, 1)
        self.ca = ChannelAttention(dim)
        self.sa = SpatialAttention()
        self.fc_a = nn.Sequential(
            nn.Linear(dim, dim // 16),
            nn.ReLU(),
            nn.Linear(dim // 16, dim),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        x = self.pres(x)
        y = self.dres(y)
        #fuse融合
        out1 = torch.cat([x, y], dim=1)  # 默认从通道上拼接: dim=0拼接后的维度是[128, 512, 7, 7]报错，应该是dim=1拼接后的维度是[64, 1024, 7, 7] (这里的64是因为GPU数量为2即batchsize一分为二分在不同的GPU上了)
        #out1 = self.ca(out1) * out1  # 取权重
        #out1 = self.sa(out1) * out1
        out1 = self.avgp(out1)  # [batch_size, 1024, 1, 1]
        out1 = out1.view(out1.size(0), -1)  # 全连接层的输入必须是二维的张量，这里是把shape转换成【batch_size, 1024】
        out1 = self.fc_a(out1) * out1
        out1 = self.fc1(out1)

        #print(out1.shape)
        #out1 = self.fc1(out1)*out1

        # 求余弦相似度
        #out2 = F.cosine_similarity(x, y, eps=1e-08)  # 64*7*7
        #out2 = cosine_similarity_onnx_exportable(x, y)  # 64*7
        '''out2 = self.avgp(out2)  # 64*1*1
        out2 = out2.view(-1, 1)
        print(out2.shape)
        #out2 = out2.mean(dim=1, keepdim=True)  # 512*1
        out2 = self.fc2(out2)  # 1
        # out2 = F.cosine_similarity(x.reshape(1, -1), y.reshape(1, -1), dim=1, eps=1e-08)
        #return self.relu(out1), 1-self.relu(out2)  两种方式的输出，一种是相异性一种是相似性
        #return 1-self.relu(out2)'''
        return self.relu(out1)