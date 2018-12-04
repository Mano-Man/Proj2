# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as tf
from src.util.torch import net_summary
from src.util.gen import banner
import numpy as np


# ----------------------------------------------------------------------------------------------------------------------
#                                                    NN Configurations
# ----------------------------------------------------------------------------------------------------------------------
def ResNet18Spatial(device, **kwargs):
    return ResNetS(BasicBlockS, [2, 2, 2, 2], device, **kwargs)


def ResNet34Spatial(device, **kwargs):
    return ResNetS(BasicBlockS, [3, 4, 6, 3], device, **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
#                                                    Layer Configurations
# ----------------------------------------------------------------------------------------------------------------------
class Spatial(nn.Module):
    def __init__(self, channels):
        super(Spatial, self).__init__()

        self.channels = channels
        self.ops_saved = 0
        self.total_ops = 0
        self.is_init = False
        self.enable = False

        self.mask = None

        # self.p_size, self.batch_size,self.in_shape,
        # self.p_size = None self.batch_size = None self.in_shape = None self.use_cuda = None
        # # self.conv_filt = None  self.batch_mask = None

    def init_to_input(self, p_size, batch_size, in_shape, use_cuda):

        self.p_size, self.batch_size, self.in_shape, self.use_cuda = p_size, batch_size, in_shape, use_cuda
        self.conv_filt = nn.Conv2d(self.channels, self.channels, kernel_size=self.p_size, stride=self.p_size,
                                   bias=False, groups=self.channels)
        self.conv_filt.weight.data.fill_(1)
        # Make convolution later constant on backward passes
        for p in self.conv_filt.parameters():
            p.requires_grad = False

        if use_cuda:
            self.conv_filt = self.conv_filt.cuda()

        self.pad_s = in_shape[1] % self.p_size  # Using pad_s as res to save complexity
        if self.pad_s != 0:
            self.pad_s = self.p_size - self.pad_s

        self.total_ops_to_add_by = batch_size * in_shape[0] * in_shape[1] * in_shape[2]

        self.is_init = True

    def set_constant_mask(self, val):
        # To be used only after layer is initialized to input size
        assert self.is_init  # Did not include this in set_mask to savee complexity
        self.set_mask(val * torch.ones(self.in_shape))

    def set_mask(self, mask):
        # To be used only after layer is initialized to input size
        self.mask = mask
        self.batch_mask = self.mask.unsqueeze(0).repeat(self.batch_size, 1, 1, 1)
        if self.use_cuda:
            self.mask = self.mask.cuda()  # TODO - Check if we can already allocate on GPU, maybe with set_
            self.batch_mask = self.batch_mask.cuda()

    def set_enable(self, enable):
        # To be used only after layer is initialized to input size
        self.enable = enable

    def reset_ops(self):
        self.total_ops = 0
        self.ops_saved = 0

    def forward(self, x):
        if not self.enable:
            return x

        if self.pad_s != 0:  # TODO - Check this works - Is this a CUDA vector?
            x = torch.nn.functional.pad(x, (0, self.pad_s, 0, self.pad_s), value=0)  # Pad with ZEROS
            assert False

        if x.size(0) != self.batch_size:
            # print('Batch size event') - Will happen once every test forward
            batch_mask = self.batch_mask[:x.size(0), :, :, :]
        else:
            batch_mask = self.batch_mask

        # The convolution basically sums over all non-zero cells. We get a block predicator for each patch
        # Anywhere that is zero - That's where we are saving operations
        b = (self.conv_filt(torch.mul(x, batch_mask)) > 0).float()
        # TODO - Check if this could be shortened (Maybe via the Pytorch Upsample module?)
        b_expanded = b.repeat(1, 1, self.p_size, self.p_size). \
            reshape(b.size(0), b.size(1), self.p_size, -1).permute(0, 1, 3, 2). \
            reshape(b.size(0), b.size(1), b.size(2) * self.p_size, -1)

        self.ops_saved += torch.sum(torch.mul(1 - b_expanded, 1 - batch_mask))
        self.total_ops += self.total_ops_to_add_by  # Could probably be calculated in advance
        # Out predicator (after padding removal)
        if self.pad_s == 0:
            return torch.mul(x, b_expanded)
        else:
            return torch.mul(x, b_expanded)[:, :, :x.size(2), :x.size(3)]  # TODO - Check this works


# ----------------------------------------------------------------------------------------------------------------------
#                                                    NN Base
# ----------------------------------------------------------------------------------------------------------------------
class ResNetS(nn.Module):  # TODO - Add a prototype "Spatial" to this - so it must implement update_spatial
    def __init__(self, block, blocks_per_layer, device, num_classes=10):
        super(ResNetS, self).__init__()

        # ResNet Definitions:
        self.in_planes = 64
        self.blocks_per_layer = blocks_per_layer
        self.device = device
        self.use_cuda = True if str(self.device) == "cuda" else False

        # Net Structure: Spatial layers are turned off by default
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pred = Spatial(64)
        self.layer1 = self._populate_block(block, 64, blocks_per_layer[0], stride=1)
        self.layer2 = self._populate_block(block, 128, blocks_per_layer[1], stride=2)
        self.layer3 = self._populate_block(block, 256, blocks_per_layer[2], stride=2)
        self.layer4 = self._populate_block(block, 512, blocks_per_layer[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.spatial_layers = [self.pred]
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                self.spatial_layers.append(block.pred1)
                self.spatial_layers.append(block.pred2)
        self.spatial_params = None  # Will be set on first init of spatial layers - is a tuple

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.relu(out)
        out = self.pred(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = tf.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def summary(self,x_shape,print_it=True):

        if self.spatial_params is None: #Layers were never init
            return net_summary(self, x_shape, device=str(self.device), print_it=print_it)
        else:
            enabled = self.enabled_layers()
            self.disable_spatial_layers(range(self.num_spatial_layers()))
            summary = net_summary(self, x_shape, device=str(self.device), print_it=print_it)
            self.enable_spatial_layers(enabled)
            return summary

    def reset_ops(self):
        for sp in self.spatial_layers:
            sp.reset_ops()

    def num_ops(self):

        ops_saved = 0
        total_ops = 0
        for sp in self.spatial_layers:
            ops_saved += int(sp.ops_saved)
            total_ops += sp.total_ops
        return ops_saved, total_ops

    def print_ops_summary(self):
        for i, sp in enumerate(self.spatial_layers):
            if sp.total_ops == 0:
                print(f'Spatial Layer {i}: Ops saved: {sp.ops_saved}/{sp.total_ops}')
            else:
                print(
                    f'Spatial Layer {i}: Ops saved: {sp.ops_saved/sp.total_ops:.3f} [{int(sp.ops_saved)}/{sp.total_ops}]')

    def initialize_spatial_layers(self, x_shape, batch_size, p_size):

        self.spatial_params = None # Destroy any reminders - if we are possibly using this net on another dataset. 
        # From init phase and on, set the spatial sizes so it won't affect the total ops.
        self.spatial_params = self.generate_spatial_sizes(x_shape)

        for i, layer in enumerate(self.spatial_layers):
            layer.init_to_input(p_size=p_size, batch_size=batch_size, in_shape=self.spatial_params[i],
                                use_cuda=self.use_cuda)

    def num_spatial_layers(self):
        return len(self.spatial_layers)

    def enabled_layers(self):
        return [i for i in range(self.num_spatial_layers()) if self.spatial_layers[i].enable]

    def disabled_layers(self):
        return [i for i in range(self.num_spatial_layers()) if not self.spatial_layers[i].enable]

    def enable_spatial_layers(self, idx_list):
        for resurrected in idx_list:
            self.spatial_layers[resurrected].set_enable(True)

    def disable_spatial_layers(self,idx_list):
        for goner_id in idx_list:
            self.spatial_layers[goner_id].set_enable(False)

    def print_spatial_status(self):

        init_status = ['-Initialized-' if sp.is_init else '-Uninitialized-' for sp in self.spatial_layers]
        enable_status = ['-Enabled-' if sp.enable else '-Disabled-' for sp in self.spatial_layers]
        mask_status = ['-Mask Not Set-' if sp.mask is None else '-Mask Set-' for sp in self.spatial_layers]
        banner('Spatial Status')
        for i, (iS, eS, mS) in enumerate(zip(init_status, enable_status, mask_status)):
            print(f'Spatial Layer {i}: {iS} {eS} {mS}')

    def strict_mask_update(self, update_ids, masks):
        # Turn on all the update ids
        self.lazy_mask_update(update_ids, masks)

        # Turn off all others
        disabled = [i for i in range(len(self.spatial_layers)) if i not in update_ids]
        self.disable_spatial_layers(disabled)


    def lazy_mask_update(self, update_ids, masks):
        for (i, mask) in zip(update_ids, masks):
            self.spatial_layers[i].set_mask(mask)
            self.spatial_layers[i].set_enable(True)

    def fill_masks_to_val(self, val):
        for layer in self.spatial_layers:
            layer.set_constant_mask(val)
            layer.set_enable(True)

    def generate_spatial_sizes(self, x_shape):

        if self.spatial_params is None:  # Spatial layers were not init, so no problem with ops
            summary = net_summary(self, x_shape, device=str(self.device), print_it=False)
            spatial_params = tuple(tuple(value['input_shape'][1:]) for key, value in summary.items() if key.startswith('Spatial'))
            assert len(spatial_params) == self.num_spatial_layers()
            return spatial_params
        else:
            return self.spatial_params

    def _populate_block(self, block, planes, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


class BasicBlockS(nn.Module):
    # Static Variable
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockS, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.pred1 = Spatial(planes)
        self.pred2 = Spatial(planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = tf.relu(self.bn1(self.conv1(x)))
        out = self.pred1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.relu(out)
        out = self.pred2(out)
        return out
