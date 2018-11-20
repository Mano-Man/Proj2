# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as tf
# ----------------------------------------------------------------------------------------------------------------------
#                                                    NN Configurations
# ----------------------------------------------------------------------------------------------------------------------
def ResNet18Spatial(sp_list, **kwargs):
    return ResNetS(BasicBlockS, [2, 2, 2, 2], sp_list, **kwargs)


def ResNet34(**kwargs):
    return ResNetS(BasicBlockS, [3, 4, 6, 3], **kwargs)


# ----------------------------------------------------------------------------------------------------------------------
#                                                    Layer Configurations
# ----------------------------------------------------------------------------------------------------------------------

class Spatial(nn.Module):
    def __init__(self, channels, spatial_params):
        super(Spatial, self).__init__()

        self.enable, self.filt_size, self.mask = spatial_params  # Presuming square filter

        self.conv_filt = nn.Conv2d(channels, channels, kernel_size=self.filt_size, stride=self.filt_size,
                                   bias=False)
        self.conv_filt.weight.data.fill_(1)
        self.ops_saved = 0
        self.total_ops = 0

        # Make convolution later constant on backward passes
        for p in self.conv_filt.parameters():
            p.requires_grad = False

        if torch.cuda.is_available():  # Will not work for Multiple GPUs
            self.mask = self.mask.cuda()

    def forward(self, x):

        if self.enable:
            # Handle input padding: (Not needed for patch_size = 2x2 on CIFAR 10)
            pad_s = x.size(2) % self.filt_size
            x_padded = torch.nn.functional.pad(x, (0, pad_s, 0, pad_s), value=0)  # Pad with ZEROS

            # Destroy anything on mask 0:
            # x.size(0) is batch size I think - This can be taken off the calculation if we truncate to BATCH_SIZE data length
            batch_mask = self.mask.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
            x_padded = torch.mul(x_padded, batch_mask)

            # The convolution basically sums over all non-zero cells. We get a block predicator for each patch
            b = (self.conv_filt(x_padded) > 0).float()

            # TODO - Check if this could be shortened
            pred_mask_ex = b.repeat(1, 1, self.filt_size, self.filt_size). \
                reshape(b.size(0), b.size(1), self.filt_size, -1).permute(0, 1, 3, 2). \
                reshape(b.size(0), b.size(1), b.size(2) * self.filt_size, -1)

            # TODO: Might be problematic saving this on all runs.
            self.ops_saved += torch.sum(torch.mul(1 - pred_mask_ex, 1 - batch_mask))  # TODO - The -1 might suffer from numeric error
            self.total_ops += x.size(0) * x.size(1) * x.size(2) * x.size(3)  # Probably the fastest way to do this
            # Out predicator (after padding removal):
            return torch.mul(x, pred_mask_ex[:, :, 0:x.size(2), 0:x.size(3)])

        else:
            return x


# ----------------------------------------------------------------------------------------------------------------------
#                                                    Layer Configurations
# ----------------------------------------------------------------------------------------------------------------------
# def expand_mat_gpu(x, p_size):
#     x_mask = x.repeat(1, 1, p_size, p_size).reshape(x.size(0), x.size(1), p_size, -1).permute(0, 1, 3, 2).reshape(
#         x.size(0), x.size(1), x.size(2) * p_size, -1)
#
#     return x_mask
# class Spatial(nn.Module):
#     def __init__(self, channels, spatial_params):
#         super(Spatial, self).__init__()
#         self.channels = channels
#         self.filt_size = spatial_params[0]
#         self.mask = spatial_params[1]
#         self.filt = torch.ones(self.filt_size, self.filt_size)
#         self.conv_filt = nn.Conv2d(channels, channels, kernel_size=self.filt_size, stride=self.filt_size, bias=False)
#         self.conv_filt.weight.data.fill_(1)
#         if torch.cuda.is_available():
#             self.conv_filt.cuda()
#
#     def forward(self, x):
#         self.conv_filt.weight.data.fill_(1)
#         if torch.cuda.is_available():
#             self.conv_filt.cuda()
#
#         res = x.size(2) % self.filt_size
#         if res != 0:
#             pad_s = self.filt_size - res
#         else:
#             pad_s = 0
#         x_padded = torch.nn.functional.pad(x, (0, pad_s, 0, pad_s), value=0)
#         # new line
#         x_padded = torch.mul(x_padded, self.mask.unsqueeze(0).repeat(x.size(0), 1, 1, 1))
#         pred_mask_sq = self.conv_filt(x_padded)
#
#         # simulating a prediction by masking sub-matrices within the input
#         pred_mask_sq = (pred_mask_sq > 0).to('cuda', dtype=torch.float32)
#
#         # expand
#         pred_mask_ex = expand_mat_gpu(pred_mask_sq, self.filt_size)
#         assert (pred_mask_ex.size(2) == (x.size(2) + pad_s))
#         pred_mask_ex = pred_mask_ex[:, :, 0:x.size(2), 0:x.size(3)]
#
#         # element-wise mult
#         x_pred = torch.mul(x, pred_mask_ex)
#         return x_pred


# ----------------------------------------------------------------------------------------------------------------------
#                                                    NN Base
# ----------------------------------------------------------------------------------------------------------------------
class ResNetS(nn.Module):
    def __init__(self, block, num_blocks, sp_list, num_classes=10):
        super(ResNetS, self).__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pred = Spatial(64, sp_list[0])
        self.layer1 = self._make_layer(block, 64, num_blocks[0], sp_list[1], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], sp_list[2], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], sp_list[3], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], sp_list[4], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, num_blocks, spatial_params, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, spatial_params, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def no_of_operations(self):
        ops_saved = self.pred.ops_saved + self.layer1[0].pred.ops_saved + self.layer1[1].pred.ops_saved \
                                        + self.layer2[0].pred.ops_saved + self.layer2[1].pred.ops_saved \
                                        + self.layer3[0].pred.ops_saved + self.layer3[1].pred.ops_saved \
                                        + self.layer4[0].pred.ops_saved + self.layer4[1].pred.ops_saved
                                        
        total_ops = self.pred.total_ops + self.layer1[0].pred.total_ops + self.layer1[1].pred.total_ops \
                                        + self.layer2[0].pred.total_ops + self.layer2[1].pred.total_ops \
                                        + self.layer3[0].pred.total_ops + self.layer3[1].pred.total_ops \
                                        + self.layer4[0].pred.total_ops + self.layer4[1].pred.total_ops
        
        return ops_saved, total_ops
        
        


class BasicBlockS(nn.Module):
    # Static Variable
    expansion = 1

    def __init__(self, in_planes, planes, spatial_params, stride=1):
        super(BasicBlockS, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.pred = Spatial(planes, spatial_params)
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
        out = self.pred(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = tf.relu(out)
        out = self.pred(out)
        return out

# ----------------------------------------------------------------------------------------------------------------------
#                                                    NN Base
# ----------------------------------------------------------------------------------------------------------------------
# def expand_mat(x, patch_size, final_size):
#     x_cpu = x.cpu()
#     x_numpy = x_cpu.data.numpy()
#     x_numpy = x_numpy.repeat(patch_size, 2).repeat(patch_size, 3)  # faster than kron
#     x_mask = torch.FloatTensor(x_numpy)
#
#     pad_s = final_size - x_mask.size(3)
#     if pad_s > 0:
#         x_mask = torch.nn.functional.pad(x_mask, (0, pad_s, 0, pad_s), value=1)
#         assert (1 == 0)  # in this implementation I should be here
#
#     if torch.cuda.is_available():
#         x_mask = x_mask.cuda()
#
#     return x_mask
