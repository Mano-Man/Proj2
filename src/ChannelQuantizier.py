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

class ChannelQuantizier():
    def __init__(self, rec, min_acc, patch_size, default_in_pattern=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.input = rec.gen_pattern_lists(min_acc)
        if default_in_pattern is None:
            self.default_in_pattern = np.ones((self.input_patterns.shape[0],self.input_patterns.shape[0]), dtype=self.input_patterns.dtype)
        else:
            self.default_in_pattern = default_in_pattern     
        self._generate_patterns(rec.mode)
        self.output_rec.filename = 'ChannelQ_mc'+ str(min_acc) + '_' + rec.filename
            
    def simulate(self):
        
        nn = net.NeuralNet(cfg.SP_MOCK)
        test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
        _, test_acc, _ = nn.test(test_gen)
        print(f'==> Asserted test-acc of: {test_acc}\n')
        
        save_counter = 0
        st_point = self.output_rec.find_resume_point()
        for layer in tqdm(range(st_point[0],len(cfg.LAYER_LAYOUT))):
            for p_idx in tqdm(range(st_point[3], len(self.output_rec.all_patterns[layer]))):
                st_point[3] = 0
                sp_list = cfg.SP_MOCK
                sp_list[layer] = (1, self.patch_size, torch.from_numpy(self.output_rec.all_patterns[layer][p_idx]))
                nn.update_spatial(sp_list)
                _, test_acc, _ = nn.test(test_gen)
                ops_saved, ops_total = nn.net.num_ops()
                self.output_rec.addRecord(ops_saved.item(), ops_total, test_acc, layer, 0, 0, p_idx)
    
                save_counter += 1
                if save_counter > cfg.SAVE_INTERVAL:
                    self.save_state()
                    save_counter = 0
        self.save_state()
        self.output_rec.save_to_csv(cfg.RESULTS_DIR)
            
        
    def save_state(self):
        rc.save_to_file(self.output_rec, True, cfg.RESULTS_DIR)
        
    def _generate_patterns(self, mode):                
        self.output_rec = rc.Record(0,0,False, mode, 0, None , \
                                    (len(self.input),[1]*len(self.input),[1]*len(self.input),None))
        self.output_rec.no_of_patterns, self.output_rec.all_patterns = self._gen_patterns_zip_longest()
        self.output_rec._create_results()
       
#    
    def _gen_patterns_zip_longest(self):
        all_patterns = []
        no_of_patterns = 0
        input_new = []
        no_of_patterns = [None]*len(self.input)
        for l in range(len(self.input)):
            layers = []
            input_new.append([self.input[l][c][0] for c in range(cfg.LAYER_LAYOUT[l][0])])
            for layer_opt in zip_longest(*input_new[l], fillvalue=(-1,-1,-1)):
                layer = np.ones(cfg.LAYER_LAYOUT[l], dtype=self.input_patterns.dtype)
                for idx, opt in enumerate(layer_opt):
                    if opt[0] >= 0 :
                        layer[idx,:,:] = mf.tile_opt((cfg.LAYER_LAYOUT[l][1],cfg.LAYER_LAYOUT[l][2]),self.input_patterns[:,:,opt[0]], False)
                    else:
                        layer[idx,:,:] = mf.tile_opt((cfg.LAYER_LAYOUT[l][1],cfg.LAYER_LAYOUT[l][2]),self.default_in_pattern, False)
                layers.append(layer)
            no_of_patterns[l] = len(layers)
            all_patterns.append(layers)
        self.input = input_new
        return no_of_patterns, all_patterns
    
# Test
#RESULTS_DIR = './data/results/'        
#uniform_patch_res = 'ps2_ones(1, 3)_uniform_patch_acc93.83_mg1024_17860D15h.pkl'
#rec_in = rc.load_from_file(uniform_patch_res,RESULTS_DIR)
#cQ = ChannelQuantizier(rec_in,92,2) 
