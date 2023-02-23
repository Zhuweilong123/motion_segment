import torch
import torch.nn as nn
import math
import os
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchsummary import summary

# pretrained: mobileNetV2 + coordinate attention

__all__ = ['mbv2_ca']
model_dir = '/home/zhanl/data/code/motion_seg/model_data'

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # coordAtt中平均池化的次数非常多，导致网络结构中size经常转换成[w, 1]和[1, h]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                # --------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                # --------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                # -----------------------------------#
                #   利用1x1卷积进行通道数的调整
                # -----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # -----------------------------------#
                #   利用1x1卷积进行通道数的上升
                # -----------------------------------#
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                # --------------------------------------------#
                #   进行3x3的逐层卷积，进行跨特征点的特征提取
                # --------------------------------------------#
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # coordinate attention
                CoordAtt(hidden_dim, hidden_dim),
                # pw-linear
                # -----------------------------------#
                #   利用1x1卷积进行通道数的下降
                # -----------------------------------#
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


class MBV2_CA(nn.Module):
    def __init__(self, num_classes=3, width_mult=1.):  # 最后输出[b, 3, 1, 1]作为decoder的输入
        super(MBV2_CA, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            #t是扩展因子，第一层1x1卷积层中卷积核的扩展倍率，c是输出特征矩阵深度channel，
            #n是bottleneck的重复次数，s是步距（ 针对第一层，其他为1 ，与ResNet的类似，通过第一层的步长改变尺寸变化）
            # t, c, n, s
            [1, 16, 1, 1],  # 256, 256, 32 -> 256, 256, 16
            [6, 24, 2, 2],  # 256, 256, 16 -> 128, 128, 24   2
            [6, 32, 3, 2],  # 128, 128, 24 -> 64, 64, 32     4
            [6, 64, 4, 2],  # 64, 64, 32 -> 32, 32, 64       7
            [6, 96, 3, 1],  # 32, 32, 64 -> 32, 32, 96
            [6, 160, 3, 2],  # 32, 32, 96 -> 16, 16, 160     14
            [6, 320, 1, 1],  # 16, 16, 160 -> 16, 16, 320    17
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)  # 这里和原mobilenetv2代码有些不同，最后一层卷积层没有包括在features里面

        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        ##直接使用预训练权重会报错"Missing key,unexpected key"，原因是训练时按照torch的命名规则,classifier里面这样linear层需要保存权重值,即网络参数名为classifier.1.weight(如果是eval(),则为classifier.0.weight)，但是预训练权重是训练完网络后删掉了dropout层保存的，参数名就会变成classifier.weight
        self.classifier = nn.Sequential(
                    nn.Dropout(0.1),  
                    nn.Linear(output_channel, num_classes)
                )
        #self.final_conv =  不确定是否需要conv层将num_classes转到3维
        """self.classifier = nn.Linear(output_channel, num_classes)"""

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, self.features

    def _initialize_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                # print(m.weight.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# 排查模型参数
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        print('[Warning] missing keys: {}'.format(missing_keys))
        print('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        print('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        print('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    print('used keys:{}'.format(len(used_pretrained_keys)))

    assert len(used_pretrained_keys) > 0, \
        'check_key load NONE from pretrained checkpoint'
    return True


# 输出一个list，[bs, bs, .., x%bs] 即新通道可以使用原通道数weight的次数
def make_batches(x, bs):
    '''
    Sample make_batches(11,3) = [3,3,3,2]
    '''
    if(x<=bs):
        return [min(x, bs)]
    else:
        return [bs] + make_batches(x-bs,bs)


# 构建新权重，主要是为了多出来的通道数可以利用原通道数weight
def create_new_weights(original_weights, nChannels):
    dst = torch.zeros(original_weights.shape[0], nChannels, original_weights.shape[2], original_weights.shape[3])
    # Repeat original weights up to fill dimension
    start = 0
    for i in make_batches(nChannels, 3):
        # print('dst',start,start+i, ' = src',0,i)
        dst[:, start:start + i, :, :] = original_weights[:, :i, :, :]
        start = start + i
    return dst


# 每次调用都要重新修改一次conv1的权重，是不是要直接新的模型文件保存到本地，然后直接读取更快
def mobilenetv2_ca(**kwargs):
    model = MBV2_CA(**kwargs)
    checkpoint = torch.load(os.path.join(model_dir, "mbv2_ca.pth"))  # 预训练模型的地址

    #print(list(checkpoint.keys()))  #features.0.0.weight为第一个卷积层的预训练权重名
    # print(checkpoint['features.0.0.weight'].size())

    #check_keys(model, checkpoint)
    model.load_state_dict(checkpoint, strict=False)  # 没有使用到classifier层的预训练权重

    old_conv1 = model.features[0][0]
    new_conv1 = nn.Conv2d(
        in_channels=old_conv1.in_channels + 3,  # 改成适合自己任务的通道数，此处通道数为 3+1=4
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=True if old_conv1.bias else False,
    )
    tmp = model.features[0][0].weight.clone()
    new_conv1.weight = nn.Parameter(create_new_weights(tmp, 6))
    # 没有改变原本地模型只改变了当前的模型
    model.features[0][0] = new_conv1
    return model


if __name__ == "__main__":
    model = mobilenetv2_ca()
    model = model.cuda()
    print(model)
    #summary(model, (6, 224, 224))

    #print(model.state_dict().items())

    #model.load_state_dict(checkpoint)  # 将预训练参数加载到模型中，但是有key对不上一直报错，Missing key(s) in state_dict: "classifier.1.weight", "classifier.1.bias"， strict=False 就能够完美的解决这个问题。也即，与训练权重中与新构建网络中匹配层的键值就进行使用，没有的就默认初始化。

    #pretrain_dict = model.state_dict()
    #print(model.items())
    #x = torch.rand(1, 6, 128, 416)
    #low_feature = model.features[:4](x)

    #high_feature = model.features[4:](low_feature)


"""
mbv2 = mbv2_ca()
mbv2 = mbv2.cuda()

x = torch.rand(1, 3, 224, 224)
x = x.cuda()
i = mbv2(x)
print(i.shape)
summary(mbv2, (3, 224, 224))"""