# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:22:47 2018

@author: Inna
"""

import Record as rc
import Config as cfg
import NeuralNet as net
import maskfactory as mf
from util.data_import import CIFAR10_Test
from tqdm import tqdm
import numpy as np
import torch
from itertools import zip_longest

class PatchQuantizier():
    def __init__(self, rec, min_acc, patch_size, default_in_pattern=None, out_rec=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.actual_patch_sizes = rec.patch_sizes
        self.input = rec.gen_pattern_lists(min_acc)
        
        if default_in_pattern is None:
            self.default_in_pattern = np.ones((self.input_patterns.shape[0],self.input_patterns.shape[0]), dtype=self.input_patterns.dtype)
        else:
            self.default_in_pattern = default_in_pattern  
            
        if out_rec is None:
            self._generate_patterns(rec.mode)
            self.output_rec.filename = 'PatchQ_ma'+ str(min_acc) + '_' + rec.filename
        else:
            self.output_rec = out_rec
        
            
    def simulate(self):
        
        st_point = self.output_rec.find_resume_point()
        if None==st_point:
           return
       
        print('==> starting simulation. file will be saved to ' + self.output_rec.filename)
        nn = net.NeuralNet(cfg.SP_MOCK)
        test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
        _, test_acc, _ = nn.test(test_gen)
        print(f'==> Asserted test-acc of: {test_acc}\n')
        
        save_counter = 0
        for l in tqdm(range(st_point[0],len(cfg.LAYER_LAYOUT))):
            for c in tqdm(range(st_point[1],len(self.output_rec.all_patterns[l]))):
                st_point[1]=0
                for p_idx in tqdm(range(st_point[3], len(self.output_rec.all_patterns[l][c]))):
                    st_point[3] = 0
                    sp_list = cfg.SP_MOCK
                    if self.output_rec.mode == rc.uniform_filters:
                        mask = mf.tile_opt(cfg.LAYER_LAYOUT[l], self.output_rec.all_patterns[l][c][p_idx], True)
                    else:
                        mask = np.ones(cfg.LAYER_LAYOUT[l], dtype = self.default_in_pattern.dtype)
                    sp_list[l] = (1, self.patch_size, torch.from_numpy(mask))
                    nn.update_spatial(sp_list)
                    _, test_acc, _ = nn.test(test_gen)
                    ops_saved, ops_total = nn.net.num_ops()
                    self.output_rec.addRecord(ops_saved.item(), ops_total, test_acc, l, c, 0, p_idx)
        
                    save_counter += 1
                    if save_counter > cfg.SAVE_INTERVAL:
                        self.save_state()
                        save_counter = 0
        self.output_rec.save_to_csv(cfg.RESULTS_DIR)
        self.output_rec.fill_empty()
        self.save_state()
        
        print('==> finised PatchQuantizier simulation.')    
        
    def save_state(self):
        rc.save_to_file(self.output_rec, True, cfg.RESULTS_DIR)
        
    def _generate_patterns(self, mode):                
        self.output_rec = rc.Record(0,0,False, mode, 0, None , \
                                    (len(self.input),[1]*len(self.input),[1]*len(self.input),None))
        self.output_rec.no_of_patterns, self.output_rec.all_patterns = self._gen_patterns_zip_longest()
        self.output_rec.no_of_channels = [len(c) for c in self.output_rec.all_patterns]
        self.output_rec._create_results()
       
#    
    def _gen_patterns_zip_longest(self):
        all_patterns = []
        no_of_patterns = [None]*len(cfg.LAYER_LAYOUT)
        for l in range(len(self.input)):
            layers = []
            no_of_patterns[l] = [0]*len(self.input[l])
            for c in range(len(self.input[l])):
                channels = []
                for channel_opt in zip_longest(*self.input[l][c], fillvalue=(-1,-1,-1)):
                    channel = np.ones((cfg.LAYER_LAYOUT[l][1], cfg.LAYER_LAYOUT[l][2]), dtype=self.input_patterns.dtype)
                    for patch_idx, opt in enumerate(channel_opt):
                        patch_n, patch_m = mf.get_patch_indexes(patch_idx, cfg.LAYER_LAYOUT[l][1], self.actual_patch_sizes[l])
                        p = self.input_patterns[:,:,opt[0]]
                        if (self.actual_patch_sizes[l] != self.patch_size):
                            p = mf.tile_opt((self.actual_patch_sizes[l],self.actual_patch_sizes[l]), p, False)
                        channel = mf.change_one_patch2d(channel, patch_n, patch_m, self.actual_patch_sizes[l], p)
                    no_of_patterns[l][c] += 1
                    channels.append(channel)
                layers.append(channels)
            all_patterns.append(layers)
        patterns_max_count = [max(no_of_patterns[l]) for l in range(len(self.input))]
        return patterns_max_count, all_patterns
    
# Test
#RESULTS_DIR = './data/results/'        
#uniform_patch_res = 'ps2_ones(1, 3)_uniform_patch_acc93.83_mg1024_17860D15h.pkl'
#rec_in = rc.load_from_file(uniform_patch_res,RESULTS_DIR)
#cQ = ChannelQuantizier(rec_in,92,2) 
