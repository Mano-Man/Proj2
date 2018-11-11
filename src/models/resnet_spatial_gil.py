'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def expand_mat(x, s, p):
    x_cpu = x.cpu()
    x_numpy = x_cpu.data.numpy()
    x_numpy = x_numpy.repeat(s, 2).repeat(s, 3) # faster than kron
    x_mask = torch.FloatTensor(x_numpy)

    pad_s = p - x_mask.size(3)
    if pad_s>0:
        x_mask = torch.nn.functional.pad(x_mask, (0, pad_s, 0, pad_s), value=1)
        assert(1==0) # in this implementation I should be here

    if torch.cuda.is_available():
        x_mask = x_mask.cuda()

    return x_mask


class spatial(nn.Module):
    def __init__(self, channels):
        super(spatial, self).__init__()
        self.channels = channels
        self.filt = torch.ones(3,3)
        self.filt_size = 3
        self.conv_filt = nn.Conv2d(channels, channels, \
                                         kernel_size=self.filt_size, stride=self.filt_size)
        self.conv_filt.weight.data.fill_(0)
        self.conv_filt.bias.data.fill_(0)
        if torch.cuda.is_available():
            self.conv_filt.cuda()

        for f in range(channels):
            self.conv_filt.weight.data[f][f] = self.filt

    def forward(self, x):
        self.conv_filt.weight.data.fill_(0)
        self.conv_filt.bias.data.fill_(0)
        if torch.cuda.is_available():
            self.conv_filt.cuda()

        for f in range(self.channels):
            self.conv_filt.weight.data[f][f] = self.filt

        # getting a squeezed mask
        res = x.size(2) % self.filt_size
        if res!=0:
            pad_s = self.filt_size - res
        else:
            pad_s = 0
        x_padded = torch.nn.functional.pad(x, (0, pad_s, 0, pad_s), value=0)
        pred_mask_sq = self.conv_filt(x_padded)

        # simulating a prediction by masking sub-matrices within the input
        pred_mask_sq = pred_mask_sq > 0

        # expand
        pred_mask_ex = expand_mat(pred_mask_sq, self.filt_size, x.size(3))
        assert(pred_mask_ex.size(2) == (x.size(2)+pad_s))
        pred_mask_ex = pred_mask_ex[:, :, 0:x.size(2), 0:x.size(3)]

        # element-wise mult
        x_pred = torch.mul(x, pred_mask_ex)
        return x_pred


class BasicBlockSG(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockSG, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.pred = spatial(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pred(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.pred(out)
        return out




class ResNetSG(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetSG, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pred = spatial(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pred(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18SpatialGil(pretrained=False, **kwargs):
    model = ResNetSG(BasicBlockSG, [2,2,2,2], **kwargs)
    
    for p in model.pred.conv_filt.parameters():
        p.requires_grad = False
    for p in model.layer1[0].pred.conv_filt.parameters():
        p.requires_grad = False
    for p in model.layer1[1].pred.conv_filt.parameters():
        p.requires_grad = False
    for p in model.layer2[0].pred.conv_filt.parameters():
        p.requires_grad = False
    for p in model.layer2[1].pred.conv_filt.parameters():
        p.requires_grad = False
    for p in model.layer3[0].pred.conv_filt.parameters():
        p.requires_grad = False
    for p in model.layer3[1].pred.conv_filt.parameters():
        p.requires_grad = False
    for p in model.layer4[0].pred.conv_filt.parameters():
        p.requires_grad = False
    for p in model.layer4[1].pred.conv_filt.parameters():
        p.requires_grad = False


    #if pretrained:
    #   model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), False)


    return model


# test()
