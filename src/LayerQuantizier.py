# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:51:55 2018

@author: Inna
"""
import Record as rc
import Config as cfg
import NeuralNet as net
import maskfactory as mf
from util.data_import import CIFAR10_Test,CIFAR10_shape
from tqdm import tqdm
import numpy as np
import torch
from functools import reduce
from itertools import product

class LayerQuantizier():
    def __init__(self, rec, min_acc, patch_size, default_in_pattern=None, out_rec=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.min_acc = min_acc
        
        self.input = rec.gen_pattern_lists(self.min_acc)
        self.input = [self.input[l][0][0] for l in range(len(self.input))] 
        
        if default_in_pattern is not None:
            self.default_in_pattern = default_in_pattern
        elif rec.mode == rc.uniform_layer: 
            self.default_in_pattern = np.ones((self.input_patterns.shape[0],self.input_patterns.shape[0]), dtype=self.input_patterns.dtype)
        elif rec.mode == rc.uniform_filters:
            self.default_in_pattern = np.ones((1,1), dtype=self.input_patterns[0][0][0].dtype)
        else:
            self.default_in_pattern = np.ones((1,1), dtype=self.input_patterns[0][0].dtype)
            
            
        if out_rec is None:
            self._generate_patterns(rec.mode, rec.layers_layout)
            self.output_rec.filename = 'LayerQ_ma'+ str(min_acc) + '_' + rec.filename
        else:
            self.output_rec = out_rec
        
    def simulate(self):
        
        st_point = self.output_rec.find_resume_point()
        if None==st_point:
           return
       
        print('==> starting simulation. file will be saved to ' + self.output_rec.filename)
        
        nn = net.NeuralNet()
        nn.net.initialize_spatial_layers(CIFAR10_shape(), cfg.BATCH_SIZE, cfg.PS)
        test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
        _, test_acc, _ = nn.test(test_gen)
        print(f'==> Asserted test-acc of: {test_acc}\n')
        if len(self.output_rec.all_patterns) > cfg.MAX_POSSIBILITIES:
            print(f'==> Too much possibilities! {len(self.output_rec.all_patterns)} is too much options')
            print(f'==> Run will stop when accuracy level of {self.min_acc} is achieved!')
        save_counter = 0
        for p_idx in tqdm(range(st_point[3],self.output_rec.no_of_patterns[0])):
            sp_list = []
            for l in range(len(self.input)):
                sp_list.append((1, self.patch_size, torch.from_numpy(self.output_rec.all_patterns[p_idx][l])))
            nn.net.strict_mask_update(update_ids=list(range(len(self.output_rec.layers_layout))), masks=sp_list)
            _, test_acc, _ = nn.test(test_gen)
            ops_saved, ops_total = nn.net.num_ops()
            self.output_rec.addRecord(ops_saved, ops_total, test_acc, 0, 0, 0, p_idx)
            
            if len(self.output_rec.all_patterns) > cfg.MAX_POSSIBILITIES and test_acc >= self.min_acc:
                break

            save_counter += 1
            if save_counter > cfg.SAVE_INTERVAL:
                self.save_state()
                save_counter = 0
        self.save_state()
        self.output_rec.save_to_csv(cfg.RESULTS_DIR)
        print('==> finised LayerQuantizier simulation.')
            
        
    def save_state(self):
        rc.save_to_file(self.output_rec, True, cfg.RESULTS_DIR)
        
    def _clean_input(self):
        for l in range(len(self.input)):
            self.curr_acc_th = 0
            self.input[l][:] = [tup for tup in self.input[l] if self._determine(tup)]
        del self.curr_acc_th
    
    def _determine(self, tup):
        p_idx, ops_saved, acc = tup
        if ops_saved == 0:
            return False
        elif self.curr_acc_th > acc and acc > 0:
            return False
        else:
            self.curr_acc_th = acc
        return True 
        
    def _generate_patterns(self, mode, layers_layout):
        all_possibilities = reduce(lambda x, y: x*y, [len(self.input[i]) for i in range(len(self.input))])
        if all_possibilities > cfg.MAX_POSSIBILITIES:
            self._clean_input()
        self.output_rec = rc.Record(layers_layout,0,False, mode, 0, self._gen_all_possible_patterns(mode, layers_layout) , \
                                    (1,[1],[1],0))
        self.output_rec.no_of_patterns = [len(self.output_rec.all_patterns)]
        self.output_rec._create_results()
        #else:
            #assert False, f'Too much possibilities! {len(self.output_rec.all_patterns)} is too much options'
    
    def _gen_all_possible_patterns(self, mode, layers_layout):
        all_patterns = []
        for net_opt in product(*self.input):
            pattern = [None]*len(self.input)
            for idx, opt in enumerate(net_opt):
                if opt[0] == -1:
                    pattern[idx] = mf.tile_opt(layers_layout[idx], self.default_in_pattern)
                elif mode == rc.uniform_layer:
                    pattern[idx] = mf.tile_opt(layers_layout[idx], self.input_patterns[:,:,opt[0]])
                elif mode == rc.uniform_filters:
                    pattern[idx] = mf.tile_opt(layers_layout[idx],self.input_patterns[idx][0][opt[0]])
                else: 
                    pattern[idx] = self.input_patterns[idx][opt[0]]
            all_patterns.append(pattern)
        return all_patterns
    
    def _max_opts_foreach_layer(self, max_opt_per_layer): #takes first options?
        pass
    

    
    
            
            
            

    
        
        
    
# Test
#RESULTS_DIR = './data/results/'        
#uniform_layer_res = 'ps2_ones(1, 3)_uniform_layer_acc93.83_mg1024_17860D14h.pkl'
#rec_in = rc.load_from_file(uniform_layer_res,RESULTS_DIR)
#rec_in.no_of_patterns = rec_in.all_patterns.shape[2]        
#lQ = LayerQuantizier(rec_in,92,2) 
# 
#lQ_rec =  rc.load_from_file('LayerQuantizier_ps2_ones(1, 3)_uniform_layer_acc93.83_mg1024_17860D14h.pkl', RESULTS_DIR)    
#print_best_results(lQ_rec, 92)
#cQ_rec = rc.load_from_file('ChannelQ_min_acc92.0_ps2_ones(1, 3)_uniform_patch_acc93.83_mg1024_17860D15h.pkl', RESULTS_DIR)
#lQ = LayerQuantizier(cQ_rec,92,2)