# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as tf
import numpy as np
from .SpatialNet import Spatial, SpatialNet

NORMAL = 0
UNI_BLOCK = 1
UNI_CLUSTER = 2


# ----------------------------------------------------------------------------------------------------------------------
#                                                    NN Spatial Configurations
# ----------------------------------------------------------------------------------------------------------------------
def ResNet18Spatial(device, output_channels, input_channels, data_shape):  # 17 Spatial Layers
    return ResNetS(BasicBlockS, [2, 2, 2, 2], device, output_channels, input_channels, data_shape, NORMAL)


def ResNet18SpatialUniBlock(device, output_channels, input_channels, data_shape):  # 5 Spatial Layers
    return ResNetS(BasicBlockS, [2, 2, 2, 2], device, output_channels, input_channels, data_shape, UNI_BLOCK)


def ResNet18SpatialUniCluster(device, output_channels, input_channels, data_shape):  # 8 Spatial Layers
    return ResNetS(BasicBlockS, [2, 2, 2, 2], device, output_channels, input_channels, data_shape, UNI_CLUSTER)


def ResNet34Spatial(device, output_channels, input_channels, data_shape):
    return ResNetS(BasicBlockS, [3, 4, 6, 3], device, output_channels, input_channels, data_shape, NORMAL)


# ----------------------------------------------------------------------------------------------------------------------
#                                           Generic Version
# ----------------------------------------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = tf.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = tf.relu(self.bn1(self.conv1(x)))
        out = tf.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = tf.relu(out)
        return out


# ----------------------------------------------------------------------------------------------------------------------
#                                           Expensive Spatial Version
# ----------------------------------------------------------------------------------------------------------------------
class BasicBlockS(BasicBlock):

    def __init__(self, in_planes, planes, stride=1, pred1=None, pred2=None):
        super().__init__(in_planes, planes, stride)

        self.is_uniform = True if pred1 is not None and pred1 == pred2 else False

        if pred1 is None:
            self.pred1 = Spatial(planes,3*3*in_planes)
        else:
            self.pred1 = pred1
        if pred2 is None:
            self.pred2 = Spatial(planes,3*3*planes)
        else:
            self.pred2 = pred2

    def forward(self, x):
        out = tf.relu(self.bn1(self.conv1(x)))
        out = self.pred1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.relu(out)
        out = self.pred2(out)
        return out

    def spatial_layers(self):
        if self.is_uniform:
            return [self.pred1]
        else:
            return [self.pred1, self.pred2]

    @staticmethod
    def num_sp():
        return 2


class ResNetS(SpatialNet):
    def __init__(self, block, blocks_per_layer, device, output_channels, input_channels, data_shape, spat_cfg):
        super().__init__(device)
        self.fam_name = f'ResNet{int(np.prod(blocks_per_layer)) + 2}'
        # ResNet Definitions:
        self.in_planes = 64
        self.blocks_per_layer = blocks_per_layer
        self.spat_cfg = spat_cfg

        # Net Structure: Spatial layers are turned off by default
        self.conv1 = nn.Conv2d(input_channels, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.pred = Spatial(self.in_planes, input_channels*3*3)

        self.clustering_index = 1
        self.clustering_indices = [0]  # Contains first layer

        self.layer1 = self._populate_block(block, 64, blocks_per_layer[0], stride=1)
        self.layer2 = self._populate_block(block, 128, blocks_per_layer[1], stride=2)
        self.layer3 = self._populate_block(block, 256, blocks_per_layer[2], stride=2)
        self.layer4 = self._populate_block(block, 512, blocks_per_layer[3], stride=2)

        # This is a dirty hack to fit the dense layer to any needed size.
        self.linear = lambda x: x
        linear_size = self.output_size(x_shape=data_shape, cuda_allowed=False)

        self.linear = nn.Linear(linear_size * block.expansion, output_channels)

        # TODO - Override Super variable - Find some more elegant way to do this
        self.spatial_layers = [self.pred]
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                self.spatial_layers.extend(block.spatial_layers())
                if self.spat_cfg == UNI_BLOCK:  # Take only first instance
                    break

    def forward(self, x):
        # print('Start')
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
        # print('Done')
        return out

    def _populate_block(self, block, planes, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        spat_layers_per_block = block.num_sp()

        if self.spat_cfg == UNI_BLOCK:
            spat = Spatial(planes)
            # Take only one
            self.clustering_indices.append(self.clustering_index)
        elif self.spat_cfg == UNI_CLUSTER:
            spat = None
            beg = self.clustering_index
            end = self.clustering_index + spat_layers_per_block
            self.clustering_indices.extend(list(range(beg, end)))
        elif self.spat_cfg == NORMAL:
            spat = None
            beg = self.clustering_index
            end = self.clustering_index + spat_layers_per_block * num_blocks
            self.clustering_indices.extend(list(range(beg, end)))

        self.clustering_index += num_blocks * spat_layers_per_block

        for stride in strides:
            if self.spat_cfg == UNI_CLUSTER:
                # For UniBlock case - create a layer for the current cluster
                spat = Spatial(planes)

            layers.append(block(self.in_planes, planes, stride, pred1=spat, pred2=spat))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Reguler Version
# ----------------------------------------------------------------------------------------------------------------------

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = tf.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = tf.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(*_):
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34(*_):
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(*_):
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101(*_):
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152(*_):
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == '__main__': test()
