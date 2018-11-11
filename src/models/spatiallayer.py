# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 12:36:19 2018

@author: Inna
"""
import torch
import torch.nn as nn
import math
import numpy as np


def expand_mat(x, s, p):
    x_cpu = x.cpu()
    x_numpy = x_cpu.data.numpy()
    x_numpy = x_numpy.repeat(s, 2).repeat(s, 3) # faster than kron
    x_mask = torch.FloatTensor(x_numpy)

    pad_s = p - x_mask.size(3)
    if pad_s>0:
        x_mask = torch.nn.functional.pad(x_mask, (0, pad_s, 0, pad_s), value=1)
        assert(1==0) # in this implementation I should be here

    x_mask = x_mask.cuda()

    return x_mask


class spatial(nn.Module):
    def __init__(self, channels, filt_size, mask):
        super(spatial, self).__init__()
        self.channels = channels
        self.mask = mask;
        #self.filt = torch.ones(filt_size,filt_size)
        self.filt_size = filt_size
        self.conv_filt = nn.Conv2d(channels, channels, \
                                         kernel_size=self.filt_size, stride=self.filt_size)
        self.conv_filt.weight.data.fill_(1)
        self.conv_filt.bias.data.fill_(0)
        self.conv_filt.cuda()

        
        #for f in range(channels):
        #   self.conv_filt.weight.data[f][f] = self.filt

    def forward(self, x):
        self.conv_filt.weight.data.fill_(1)
        self.conv_filt.bias.data.fill_(0)
        self.conv_filt.cuda()

        #for f in range(self.channels):
        #   self.conv_filt.weight.data[f][f] = self.filt

        # getting a squeezed mask
        res = x.size(2) % self.filt_size
        if res!=0:
            pad_s = self.filt_size - res
        else:
            pad_s = 0
        x_padded = torch.nn.functional.pad(x, (0, pad_s, 0, pad_s), value=0)
        x_padded = torch.mul(x_padded, self.mask) #new line
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