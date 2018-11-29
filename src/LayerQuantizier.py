# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:51:55 2018

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
from functools import reduce
from itertools import product

class LayerQuantizier():
    def __init__(self, rec, min_acc, patch_size, default_in_pattern=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.input = rec.gen_pattern_lists(min_acc)
        self.input = [self.input[l][0][0] for l in range(len(self.input))]
        if default_in_pattern is None:
            self.default_in_pattern = np.ones((self.input_patterns.shape[0],self.input_patterns.shape[0]), dtype=self.input_patterns.dtype)
        else:
            self.default_in_pattern = default_in_pattern     
        self._generate_patterns(rec.mode)
        self.output_rec.filename = 'LayerQ_mc'+ str(min_acc) + '_' + rec.filename
        
            
    def simulate(self):
        
        nn = net.NeuralNet(cfg.SP_MOCK)
        test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
        _, test_acc, _ = nn.test(test_gen)
        print(f'==> Asserted test-acc of: {test_acc}\n')
        
        save_counter = 0
        for p_idx in tqdm(range(self.output_rec.find_resume_point()[3],self.output_rec.no_of_patterns[0])):
            sp_list = []
            for l in range(len(self.input)):
                sp_list.append((1, self.patch_size, torch.from_numpy(self.output_rec.all_patterns[p_idx][l])))
            nn.update_spatial(sp_list)
            _, test_acc, _ = nn.test(test_gen)
            ops_saved, ops_total = nn.net.num_ops()
            self.output_rec.addRecord(ops_saved.item(), ops_total, test_acc, 0, 0, 0, p_idx)

            save_counter += 1
            if save_counter > cfg.SAVE_INTERVAL:
                self.save_state()
                save_counter = 0
        self.save_state()
        self.output_rec.save_to_csv(cfg.RESULTS_DIR)
            
        
    def save_state(self):
        rc.save_to_file(self.output_rec, True, cfg.RESULTS_DIR)
        
    def _generate_patterns(self, mode):
        all_possibilities = reduce(lambda x, y: x*y, [len(self.input[i]) for i in range(len(self.input))])
        if all_possibilities < cfg.MAX_POSSIBILITIES:
            self.output_rec = rc.Record(0,0,False, mode, 0, self._gen_all_possible_patterns() , \
                                        (1,[1],[1],0))
            self.output_rec.no_of_patterns = [len(self.output_rec.all_patterns)]
            self.output_rec._create_results()
        else:
            assert False, f'Too much possibilities! {all_possibilities} is too much options'
    
    def _gen_all_possible_patterns(self):
        all_patterns = []
        for net_opt in product(*self.input):
            pattern = [None]*len(self.input)
            for idx, opt in enumerate(net_opt):
                if opt[0] >= 0:
                    pattern[idx] = mf.tile_opt(cfg.LAYER_LAYOUT[idx], self.input_patterns[:,:,opt[0]])
                else: 
                    pattern[idx] = mf.tile_opt(cfg.LAYER_LAYOUT[idx], self.default_in_pattern)
            all_patterns.append(pattern)
        return all_patterns
            
            
            

    
        
        
    
# Test
RESULTS_DIR = './data/results/'        
#uniform_layer_res = 'ps2_ones(1, 3)_uniform_layer_acc93.83_mg1024_17860D14h.pkl'
#rec_in = rc.load_from_file(uniform_layer_res,RESULTS_DIR)
#rec_in.no_of_patterns = rec_in.all_patterns.shape[2]        
#lQ = LayerQuantizier(rec_in,92,2) 
# 
#lQ_rec =  rc.load_from_file('LayerQuantizier_ps2_ones(1, 3)_uniform_layer_acc93.83_mg1024_17860D14h.pkl', RESULTS_DIR)    
#print_best_results(lQ_rec, 92)