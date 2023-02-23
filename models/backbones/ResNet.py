# pytorch源码中的ResNet-18和Attention结合
import math
import torch
import torch.nn as nn

from models.attention.CBAM import ChannelAttention, SpatialAttention
from models.attention.coordatt import CoordAtt

'''
def cosine_similarity_onnx_exportable(x1, x2, dim=-1):
    cross = (x1 * x2).sum(dim=dim)
    x1_l2 = (x1 * x1).sum(dim=dim)
    x2_l2 = (x2 * x2).sum(dim=dim)
    return torch.div(cross, (x1_l2 * x2_l2).sqrt())'''


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    '''3x3 convolution with padding'''
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """定义BasicBlock残差块类

        参数：
            inplanes (int): 输入的Feature Map的通道数
            planes (int): 第一个卷积层输出的Feature Map的通道数
            stride (int, optional): 第一个卷积层的步长
            downsample (nn.Sequential, optional): 旁路下采样的操作
        注意：
            残差块输出的Feature Map的通道数是planes*expansion
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock_cbam(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_cbam, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)


        # 提取特征后，进行注意力计算
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # out1变成了54
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #conv2有问题变成了52
        out = self.conv2(out)
        out = self.bn2(out)


        #out = self.ca(out) * out
        #out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)  # basicblock的残差只有一个下采样层（输入输出维度相同），这里的downsample函数包含了卷积降升维和下采样（stride=2）这里的size是正确的


        out += residual
        out = self.relu(out)

        return out


class BasicBlock_coordatt(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_coordatt, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

        # 提取特征后，进行注意力计算
        self.ca = CoordAtt(planes, planes)

    def forward(self, x):
        residual = x
        # out1变成了54
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #conv2有问题变成了52
        out = self.conv2(out)
        out = self.bn2(out)


        #out = self.ca(out) * out
        #out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)  # basicblock的残差只有一个下采样层（输入输出维度相同），这里的downsample函数包含了卷积降升维和下采样（stride=2）这里的size是正确的

        out = self.ca(out)
        out += residual
        out = self.relu(out)

        return out


#######################################################################################################################
class Bottleneck(nn.Module):  # resnet18层中无bottleneck模块
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        #self.ca = ChannelAttention(planes * 4)
        #self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #out = self.ca(out) * out
        #out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)  # 升维降采样

        out += residual
        out = self.relu(out)


# 加上了通道注意力和空间注意力的块
class Bottleneck_cbam(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_cbam, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_coordatt(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_coordatt, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

        self.ca = CoordAtt(planes * self.expansion, planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)




        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.ca(out)

        out += residual
        out = self.relu(out)

        return out


#----------------------------------------------------------------------分界线----------------------------------------------------------------------------------------------------------
'''
if __name__ == '__main__':
    model = PoseNet(BasicBlock_sa, [2, 2, 2, 2])
    print(model)'''


# # 提取视觉特征512*7*7
class rgbResNet(nn.Module):
    def __int__(self, block, layers, num_classes=12):  # 输入使用的基础块类别和卷积层数
        super(rgbResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        '''self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)'''

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):  # 确定残差架构有没有出现维度不匹配的情况
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        '''x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)'''

        return x


# 直接与拼接的pose计算余弦相似度
class depthResNet(nn.Module):
    def __init__(self, block, layers, num_classes=12):
        self.inplanes = 64
        super(depthResNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 最后一个卷积层特征512*7*7，即现在的视觉特征空间
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 这个时候映射到姿态空间？
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.fc2 = nn.Linear(num_classes, 1)
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

    def forward(self, x, y):  # x是深度，y是pose

        x = self.conv1(x)  # batchsize降为了64，这里不太理解为什么batchsize会发生变化
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)  # 从深度图中提取出的特征 12维(64x12) 这里的batch_size变成了64,type=torch.cuda.FloatTensor

        output = self.cos(x, y)  # 求注意力即权重，torch.Size([64]),这里type=torch.cuda.DoubleTensor和之前的float类型不一样，因为上面余弦相似度的精度设定为1e-08
        output = output.view(-1, 1)

        output = abs(output)  # torch.Size([64])
        '''output = output.float()
        #output = output.view([1, -1])  # torch.Size([1, 64]),

        # 这里的output和x的数据类型不同报错，其中output是，x是

        output = self.fc2(torch.matmul(output, x))'''  # 加权平均：x在六维空间的特征表示乘以权重再通过一个线性层回归到一维数据  torch.Size([1])?
        #print(output.shape)

        return output  # 相似度在0~1范围内

