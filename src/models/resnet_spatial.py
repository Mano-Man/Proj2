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
    x_numpy = x_numpy.repeat(s, 2).repeat(s, 3)  # faster than kron
    x_mask = torch.cuda.FloatTensor(x_numpy)

    pad_s = p - x_mask.size(3)
    if pad_s > 0:
        x_mask = torch.nn.functional.pad(x_mask, (0, pad_s, 0, pad_s), value=1)
        assert (1 == 0)  # in this implementation I should be here

    if torch.cuda.is_available():
        x_mask = x_mask.cuda()

    return x_mask


class spatial_new(nn.Module):
    def __init__(self, channels, spatial_params):
        super(spatial_new, self).__init__()
        self.channels = channels
        self.filt_size = spatial_params[0]
        self.mask = spatial_params[1]
        self.filt = torch.ones(self.filt_size, self.filt_size)
        self.conv_filt = nn.Conv2d(channels, channels, kernel_size=self.filt_size, stride=self.filt_size)
        self.conv_filt.weight.data.fill_(1)
        self.conv_filt.bias.data.fill_(0)
        if torch.cuda.is_available():
            self.conv_filt.cuda()

        #        for f in range(channels):
        #            self.conv_filt.weight.data[f][f] = self.filt

    def forward(self, x):
        self.conv_filt.weight.data.fill_(1)
        self.conv_filt.bias.data.fill_(0)
        if torch.cuda.is_available():
            self.conv_filt.cuda()

        #        for f in range(self.channels):
        #            self.conv_filt.weight.data[f][f] = self.filt

        # getting a squeezed mask
        res = x.size(2) % self.filt_size
        if res != 0:
            pad_s = self.filt_size - res
        else:
            pad_s = 0
        x_padded = torch.nn.functional.pad(x, (0, pad_s, 0, pad_s), value=0)
        # new line
        x_padded = torch.mul(x_padded, self.mask.unsqueeze(0).repeat(x.size(0), 1, 1, 1))
        pred_mask_sq = self.conv_filt(x_padded)

        # simulating a prediction by masking sub-matrices within the input
        pred_mask_sq = pred_mask_sq > 0

        # expand
        pred_mask_ex = expand_mat(pred_mask_sq, self.filt_size, x.size(3))
        assert (pred_mask_ex.size(2) == (x.size(2) + pad_s))
        pred_mask_ex = pred_mask_ex[:, :, 0:x.size(2), 0:x.size(3)]

        # element-wise mult
        x_pred = torch.mul(x, pred_mask_ex)
        return x_pred


class BasicBlockS(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, spatial_params, stride=1):
        super(BasicBlockS, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.pred = spatial_new(planes, spatial_params)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pred(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.pred(out)
        return out


# Spatial_params_Li = (filt_size, mask)
class ResNetS(nn.Module):
    def __init__(self, block, num_blocks, sp_list, num_classes=10):
        super(ResNetS, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pred = spatial_new(64, sp_list[0])
        self.layer1 = self._make_layer(block, 64, num_blocks[0], sp_list[1], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], sp_list[2], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], sp_list[3], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], sp_list[4], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, spatial_params, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, spatial_params, stride))
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


# mask sizes for sp_i:
#   sp0[1].shape =  (64, 32, 32)
#   sp1[1].shape =  (64, 32, 32)
#   sp2[1].shape =  (128, 16, 16)
#   sp3[1].shape = (256, 8, 8)
#   sp4[1].shape = (512, 4, 4)
# sp_i = (patch_size, mask)

def ResNet18Spatial(sp_list, pretrained=False, **kwargs):
    model = ResNetS(BasicBlockS, [2, 2, 2, 2], sp_list, **kwargs)
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

    # if pretrained:
    #   model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), False)
    return model
