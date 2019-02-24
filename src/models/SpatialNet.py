# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as tf
from .PytorchNet import PytorchNet
from util.gen import banner
from math import log10,floor,ceil


# ----------------------------------------------------------------------------------------------------------------------
#                                              Spatial Layer Implementation
# ----------------------------------------------------------------------------------------------------------------------
class Spatial(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels
        self.ops_saved = 0
        self.total_ops = 0
        self.is_init = False
        self.enable = False
        self.mask = None

        self.p_size, self.batch_size, self.in_shape, = (None, None, None)
        self.padded_in_shape, self.use_cuda, self.pad_s = (None, None, None)
        self.conv_filt, self.batch_mask = (None, None)


    def init_to_input(self, p_size, batch_size, in_shape, padded_in_shape, use_cuda):

        self.p_size, self.batch_size, self.in_shape, self.padded_in_shape, self.use_cuda = \
            p_size, batch_size, in_shape, padded_in_shape, use_cuda

        self.pad_s = self.padded_in_shape[1] - self.in_shape[1]
        self.conv_filt = nn.Conv2d(self.channels, self.channels, kernel_size=self.p_size, stride=self.p_size,
                                   bias=False, groups=self.channels)
        self.conv_filt.weight.data.fill_(1)
        # Make convolution later constant on backward passes
        for p in self.conv_filt.parameters():
            p.requires_grad = False

        if use_cuda:
            self.conv_filt = self.conv_filt.cuda()

        self.is_init = True

    def set_constant_mask(self, val):
        # To be used only after layer is initialized to input size
        assert self.is_init  # Did not include this in set_mask to savee complexity
        self.set_mask(val * torch.ones(self.padded_in_shape))

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

        if self.pad_s != 0:
            x = torch.nn.functional.pad(x, (0, self.pad_s, 0, self.pad_s), value=0)  # Pad with ZEROS

        if x.size(0) != self.batch_size:
            #print('Batch size event')  # - Will happen once every test forward
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
        self.total_ops += x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        # Out predicator (after padding removal)
        if self.pad_s == 0:
            return torch.mul(x, b_expanded)
        else:
            return torch.mul(x, b_expanded)[:, :, :self.in_shape[1], :self.in_shape[1]]


# ----------------------------------------------------------------------------------------------------------------------
#                                        Spatial Net Abstract Class
# ----------------------------------------------------------------------------------------------------------------------
class SpatialNet(PytorchNet):
    def __init__(self, device):
        super().__init__(device)

        self.spatial_layers = []
        self.sp = None  # Will be set on first init of spatial layers - is a tuple
        self.sp_padded = None  # Will be set on first init of spatial layers - is a tuple
        self.x_shape = None
        self.p_size = None
        self.clustering_indices = None # Used for clustering layers together

    def forward(self, x):
        # Make it abstract
        raise NotImplementedError

    def reset_spatial(self):
        for sp in self.spatial_layers:
            sp.reset_ops()
            sp.set_enable(False)
            
    def reset_ops(self):
        for sp in self.spatial_layers:
            sp.reset_ops()

    def name(self):
        return self.__class__.__name__

    def num_ops(self):
        ops_saved = 0
        total_ops = 0
        for sp in self.spatial_layers:
            ops_saved += int(sp.ops_saved)
            total_ops += sp.total_ops
        return ops_saved, total_ops

    def print_ops_summary(self):

        all_ops_saved, all_total_ops = self.num_ops()
        for i, l in enumerate(self.spatial_layers):
            if l.total_ops == 0:
                print(f'Spatial Layer {i}: Ops saved: {l.ops_saved}/{l.total_ops}')
            else:
                spacer = ' ' * (len(str(len(self.spatial_layers))) - len(str(i)))
                assert(l.ops_saved <= all_ops_saved)
                if all_ops_saved ==0: # Handle div by 0 case
                    midstr = '0'
                else:
                    midstr = f'{l.ops_saved*100/all_ops_saved:.3f}'
                print(
                    f'Spatial Layer {i}:{spacer} Ops saved: {l.ops_saved*100 /l.total_ops:.3f} % [{int(l.ops_saved)} / {l.total_ops}]',
                    f'of [{midstr} % / {l.total_ops*100/all_total_ops:.3f} %]')

        if all_total_ops > 0:
            print(f'Grand total: {all_ops_saved}/{all_total_ops} {all_ops_saved*100/all_total_ops:.3f} %')
        else:
            print(f'Grand total: {all_ops_saved}/{all_total_ops}')

    def initialize_spatial_layers(self, x_shape, batch_size, p_size,freeze=True):

        if freeze:
            self.eval() #LOCK Network for testing only
        # From init phase and on, set the spatial sizes
        self.x_shape = x_shape
        self.p_size = p_size
        self.sp = self.generate_spatial_sizes(x_shape)
        self.sp_padded = self.generate_padded_spatial_sizes(x_shape, p_size)

        for i, layer in enumerate(self.spatial_layers):
            layer.init_to_input(p_size=p_size, batch_size=batch_size, in_shape=self.sp[i],
                                padded_in_shape=self.sp_padded[i],
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

    def disable_spatial_layers(self, idx_list):
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
        # NOTE - This returns the sizes *WITHOUT* the padding and wit
        if self.sp is None or self.x_shape != tuple(x_shape):  # Spatial layers were not init or new dataset
            summary = self.summary(x_shape, print_it=False)
            sp = tuple(tuple(value['input_shape'][1:]) for key, value in summary.items() if key.startswith('Spatial'))

            if self.clustering_indices is not None:
                sp = tuple([sp[i] for i in self.clustering_indices])
            return sp
        else:
            return self.sp

    def generate_padded_spatial_sizes(self, x_shape, p_size):

        if self.sp_padded is None or self.p_size != p_size or self.x_shape != x_shape:
            sp_padded = []
            for param in self.generate_spatial_sizes(x_shape):
                pad_s = param[1] % p_size
                if pad_s != 0:
                    pad_s = p_size - pad_s
                    sp_padded.append((param[0], param[1] + pad_s, param[2] + pad_s))
                else:
                    sp_padded.append(param)
            return tuple(sp_padded)
        else:
            return self.sp_padded

    # Override
    def summary(self, x_shape, batch_size=-1, print_it=True):
        # TODO - Make this "careful" activation wrapper a built in private function
        if self.sp is None:
            return super().summary(x_shape, print_it=print_it)
        else:
            enabled = self.enabled_layers()
            if not enabled:
                return super().summary(x_shape, print_it=print_it)
            else:
                self.disable_spatial_layers(enabled)
                summary = super().summary(x_shape, print_it=print_it)
                self.enable_spatial_layers(enabled)
                return summary

    def output_size(self, x_shape,cuda_allowed=False):

        if self.sp is None:
            return super().output_size(x_shape,cuda_allowed)
        else:
            enabled = self.enabled_layers()
            if not enabled:
                return super().output_size(x_shape,cuda_allowed)
            else:
                self.disable_spatial_layers(enabled)
                siz = super().output_size(x_shape,cuda_allowed)
                self.enable_spatial_layers(enabled)
                return siz



def sequential_spatial_layer_extract(seq_model):
    spatial_layers = []
    for idx, m in enumerate(seq_model.modules()):
        #print(idx, '->', m)
        if isinstance(m,Spatial):
            spatial_layers.append(m)
    return spatial_layers

