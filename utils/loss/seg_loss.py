import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
# pyramid loss
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=med_frq):
        super(CrossEntropyLoss2d, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(torch.from_numpy(np.array(weight)).float(),
                                           size_average=False, reduce=False)

    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            mask = targets > 0
            targets_m = targets.clone()
            targets_m[mask] -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())
            losses.append(torch.sum(torch.masked_select(loss_all, mask)) / torch.sum(mask.float()))
        total_loss = sum(losses)
        return total_loss

# 防止正负样本不平衡
def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs,
                                                                                                 temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss
    
# 一种集合相似度度量函数，通常用于计算两个样本的相似度，取值范围在[0,1]，类别少于10类的时候可以训练 
def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

"""


# 交叉熵损失# cls_weighs是是否给不同种类赋予不同的损失权值，默认是平衡的。设置的话，注意设置成numpy形式的，长度和num_classes一样。
class CELoss(nn.Module):
    def __init__(self, if_dice=False, ignore_index=255, reduction='mean', beta=1, smooth=1e-5):
        super(CELoss, self).__init__()
        self.if_dice = if_dice
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()
        # 这里的label到底该不该有num_classes通道
        nt, ht, wt = targets.size()
        if h != ht and w != wt:
            inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

        temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        temp_target = targets.view(-1)  # label要比原输入少一个维度

        CE_loss = nn.CrossEntropyLoss(ignore_index=8)(temp_inputs, temp_target)  # ignore_index表示忽略计算(梯度)的x像素值，这里设置忽略misc类

        if self.if_dice:
            # --------------------------------------------#
            #   计算dice loss
            # --------------------------------------------#
            inputs_temp = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)[..., :-1]
            targets_temp = targets.view(n, -1, 1)[..., :-1]  # 忽略最后一个标签:misc
            #tp = torch.sum(targets_temp[..., :-1] * inputs_temp, axis=[0, 1])  忽略某个标签不计算
            tp = torch.sum(targets_temp * inputs_temp, axis=[0, 1])
            fp = torch.sum(inputs_temp, axis=[0, 1]) - tp
            fn = torch.sum(targets_temp, axis=[0, 1]) - tp
            score = ((1 + self.beta ** 2) * tp + self.smooth) / (
                        (1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
            dice_loss = 1 - torch.mean(score)
            return CE_loss+dice_loss
        else:
            return CE_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()

if __name__ == "__main__":
    criterion = CELoss()
    x = torch.randn(4, 9, 128, 416)
    y = torch.ones([4, 128, 416])
    y = y.to(dtype=torch.long)
    loss = criterion(x, y)
    print()