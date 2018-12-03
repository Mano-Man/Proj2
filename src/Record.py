# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:29:28 2018

@author: Inna
"""
from bitarray import bitarray
import numpy as np
import pickle
import math
import time
import os
import csv

#Todo : add to saved files self.no_of_patterns
gran_dict = {0:"max_granularity", 1:"uniform_filters", 2:"uniform_patch", 3:"uniform_layer"}
max_granularity = 0
uniform_filters = 1
uniform_patch = 2
uniform_layer = 3


def bitmasks(n, m):
    if m < n:
        if m > 0:
            for x in bitmasks(n - 1, m - 1):
                yield bitarray([1]) + x
            for x in bitmasks(n - 1, m):
                yield bitarray([0]) + x
        else:
            yield n * bitarray('0')
    else:
        yield n * bitarray('1')


def patches(patch_size, ones_range, mask_type=np.float32):
    n = patch_size * patch_size
    for m in range(*ones_range):
        for patch in bitmasks(n, m):
            yield np.array(patch, dtype=mask_type).reshape((patch_size, patch_size))


def all_patches_array(patch_size, ones_range, mask_type=np.float32):
    all_patches = np.zeros((patch_size, patch_size, 1), dtype=mask_type)
    for p in patches(patch_size, ones_range, mask_type):
        all_patches = np.append(all_patches, p[:, :, np.newaxis], axis=2)
    return all_patches[:, :, 1:]


def actual_patch_size(N, M, patch_size, gran_thresh):
    granularity = (N * M) / (patch_size * patch_size)
    new_patch_size = patch_size
    while granularity > gran_thresh:
        new_patch_size += patch_size
        granularity = (N * M) / (new_patch_size * new_patch_size)
    return new_patch_size


class Record():
    '''
    results[layer][channel][patch][pattern] = (ops_saved, total_ops,acc)
    '''

    def __init__(self, layers_layout, gran_thresh, gen_patches, mode, initial_acc, *argv):
        self.mode = mode
        self.gran_thresh = gran_thresh
        
        if gen_patches:
            self.no_of_layers = len(layers_layout)
            self.no_of_channels = [l[0] for l in layers_layout]
            self.all_patterns = all_patches_array(argv[0], argv[1])  # [patch_size, ones_range]
            
            patch_size = self.all_patterns.shape[0]
            self.patch_sizes = [actual_patch_size(l[1], l[2], patch_size, gran_thresh) for l in layers_layout]
            self.no_of_patches = [math.ceil(layers_layout[idx][1] / self.patch_sizes[idx]) * \
                                  math.ceil(layers_layout[idx][2] / self.patch_sizes[idx]) for idx in
                                  range(self.no_of_layers)]
    
            if mode == uniform_filters:
                self.no_of_channels = [1] * self.no_of_layers
            elif mode == uniform_patch:
                self.no_of_patches = [1] * self.no_of_layers
            elif mode == uniform_layer:
                self.no_of_channels = [1] * self.no_of_layers
                self.no_of_patches = [1] * self.no_of_layers
            self.no_of_patterns = [self.all_patterns.shape[2]]*self.no_of_layers
            self._create_results()
        else:
            self.all_patterns = argv[0]
            self.no_of_layers,self.no_of_channels, self.no_of_patches, self.no_of_patterns = argv[1]

        self.filename = gran_dict[self.mode]+ '_acc' + str(initial_acc) + '_mg' + str(gran_thresh) + '_' + str(int(time.time()))
    
    def _create_results(self):
        self.results = []
        for l in range(self.no_of_layers):
            layer = []
            for k in range(self.no_of_channels[l]):
                channel = []
                for j in range(self.no_of_patches[l]):
                    patch = []
                    for i in range(self.no_of_patterns[l]):
                        patch.append(None)
                    channel.append(patch)
                layer.append(channel)
            self.results.append(layer)
            
    def indexed_according_to_mode(self,layer, channel, patch_idx, pattern_idx):
        if self.mode == uniform_filters:
            channel = 0
        elif self.mode == uniform_patch:
            patch_idx = 0
        elif self.mode == uniform_layer:
            channel = 0
            patch_idx = 0
        return layer, channel, patch_idx, pattern_idx

    def addRecord(self, op, tot_op, acc, layer, channel=0, patch_idx=0, pattern_idx=0):
        layer, channel, patch_idx, pattern_idx = self.indexed_according_to_mode(layer, channel, patch_idx, pattern_idx)
        self.results[layer][channel][patch_idx][pattern_idx] = (op, tot_op, acc)

    def find_resume_point(self):
        for layer in range(self.no_of_layers):
            for channel in range(self.no_of_channels[layer]):
                for patch_idx in range(self.no_of_patches[layer]):
                    for pattern_idx in range(self.no_of_patterns[layer]):
                        if self.results[layer][channel][patch_idx][pattern_idx] is None:
                            return [layer, channel, patch_idx, pattern_idx]
    def fill_empty(self):
        for layer in range(self.no_of_layers):
            for channel in range(self.no_of_channels[layer]):
                for patch_idx in range(self.no_of_patches[layer]):
                    for pattern_idx in range(self.no_of_patterns[layer]):
                        if self.results[layer][channel][patch_idx][pattern_idx] is None:
                            self.results[layer][channel][patch_idx][pattern_idx] = (-1,-1,-1)
                        
    def is_full(self):
        return None==self.find_resume_point()

    def save_to_csv(self, path='./data/results'):
        out_path = os.path.join(path, self.filename + ".csv")
        with open(out_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['layer', 'channel', 'patch number', 'pattern number', 'operations saved', 'total operations',
                 'accuracy'])
            for l in range(self.no_of_layers):
                for k in range(self.no_of_channels[l]):
                    for j in range(self.no_of_patches[l]):
                        for i in range(self.no_of_patterns[l]):
                            if self.results[l][k][j][i] is not None:
                                csv.writer(f).writerow([l, k, j, i, self.results[l][k][j][i][0], \
                                                        self.results[l][k][j][i][1], \
                                                        self.results[l][k][j][i][2]])
            
    def gen_pattern_lists(self, min_acc):
        slresults = []
        for l in range(self.no_of_layers):
            layer = []
            for k in range(self.no_of_channels[l]):
                channel = []
                for j in range(self.no_of_patches[l]):
                    patch = []
                    for p_idx, res_tuple in sorted(enumerate(self.results[l][k][j][:]),key=lambda x:(x[1][0],x[1][2]),  reverse=True):
                        if res_tuple[2] > min_acc:
                            patch.append((p_idx,res_tuple[0],res_tuple[2]))
                    patch.append((-1,-1,-1))
                    channel.append(patch)
                layer.append(channel)
            slresults.append(layer)
        return slresults
    
class FinalResultRc():
    def __init__(self, f_acc, ops_saved, tot_ops, mode, pattern,ps,max_acc_loss, ones_range, net_name):
        self.filename = f'FR_{net_name}_ps{ps}_ones{ones_range}_{gran_dict[mode]}_ma{max_acc_loss}_os{round((ops_saved/tot_ops)*100, 3)}_fa{f_acc}'
        self.mask = pattern
        self.final_acc = f_acc
        self.ops_saved = ops_saved
        self.total_ops = tot_ops
        self.mode = mode
        self.min_acc = max_acc_loss
        self.patch_size = ps
        self.ones_range = ones_range
        self.network = net_name


def save_to_file(record, use_default=True, path='./data/results', filename=''):
    if use_default:
        filename = record.filename
    if not os.path.isdir(path):
        os.mkdir(path)
    outfile = open(os.path.join(path, filename + '.pkl'), 'wb')
    pickle.dump(record, outfile)
    outfile.close()


def load_from_file(filename, path='./data/results'):
    infile = open(os.path.join(path, filename), 'rb')
    record = pickle.load(infile)
    infile.close()
    return record

# -----------------Test---------------------
# max_gra = 32*32
# r = Record(Resnet18_layers_layout,max_gra,True, max_granularity,2,2,3)
#
# ap = all_patches_array(2,2,3)
# r1 = Record(Resnet18_layers_layout,max_gra,False, max_granularity,ap)
#
# assert(r1.results ==r.results)
#
# r2 = Record(Resnet18_layers_layout,max_gra,True, uniform_filters,2,2,3)
# r2.addRecord( 4, 0.98, 0, 0, 0, 0)
# r2.save_to_csv()
#           
# save_to_file(r2)
# r3 = load_from_file('uniform_filters_1542549224_mg1024')
# assert(r2.results ==r3.results)
