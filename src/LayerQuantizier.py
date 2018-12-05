# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:51:55 2018

@author: Inna
"""
import numpy as np
import torch

from Record import Mode, FinalResultRc, load_from_file, save_to_file
import Config as cfg
import NeuralNet as net
import RecordFinder as rf
import maskfactory as mf
from util.data_import import CIFAR10_Test,CIFAR10_shape


class LayerQuantizier():
    def __init__(self, rec, min_acc, patch_size, default_in_pattern=None, out_rec=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.min_acc = min_acc
        self.mode = rec.mode
        self.layers_layout = rec.layers_layout
        
        self.input = rec.gen_pattern_lists(self.min_acc)
        self.input = [self.input[l][0][0] for l in range(len(self.input))] 
        self._clean_input()
        
        if default_in_pattern is not None:
            self.default_in_pattern = default_in_pattern
        elif rec.mode == Mode.UNIFORM_LAYER: 
            self.default_in_pattern = np.ones((self.input_patterns.shape[0],self.input_patterns.shape[0]), dtype=self.input_patterns.dtype)
        elif rec.mode == Mode.UNIFORM_FILTERS:
            self.default_in_pattern = np.ones((1,1), dtype=self.input_patterns[0][0][0].dtype)
        else:
            self.default_in_pattern = np.ones((1,1), dtype=self.input_patterns[0][0].dtype)
            
        self.resume_param_filename = 'RP_LayerQ_ma'+ str(min_acc) + '_' + rec.filename
        resume_param_path = rf.find_rec_filename(rec.mode,rf.lQ_RESUME)
        if resume_param_path is None:
            self.resume_index = [0]*len(self.input)
            self.is_final = [-1]*len(self.input)
        else:
            self.resume_index, self.is_final = load_from_file(resume_param_path, path='')

    def simulate(self):
        if self._is_finised():
            return
        print('==> starting LayerQuantizier simulation.')
        
        nn = net.NeuralNet()
        test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
        _, test_acc, _ = nn.test(test_gen)
        print(f'==> Asserted test-acc of: {test_acc}\n')
        
        nn.net.initialize_spatial_layers(CIFAR10_shape(), cfg.BATCH_SIZE, cfg.PS)
        _, test_acc, _ = nn.test(test_gen)
        print(f'==> Asserted test-acc of: {test_acc}\n')
        
        self.sp_list = [None]*len(self.input)
        for l_idx, p_idx in enumerate(self.resume_index):
            self._update_layer( l_idx, p_idx)
                
        nn.net.strict_mask_update(update_ids=list(range(len(self.layers_layout))), masks=self.sp_list)
        _, test_acc, _ = nn.test(test_gen)
        ops_saved, ops_total = nn.net.num_ops()
        if test_acc >= self.min_acc:
            f_rec = self._save_final_rec(test_acc,ops_saved, ops_total)
            return f_rec
        
        l_to_inc = self._get_next_opt()
        while(l_to_inc is not None):
            self.save_state()
            self._update_layer(l_to_inc, self.resume_index[l_to_inc])
                
            nn.net.lazy_mask_update(update_ids=[l_to_inc], masks=[self.sp_list[l_to_inc]])
            _, test_acc, _ = nn.test(test_gen)
            ops_saved, ops_total = nn.net.num_ops()
            if test_acc >= self.min_acc:
                f_rec = self._save_final_rec(test_acc,ops_saved, ops_total)
                return f_rec
            l_to_inc = self._get_next_opt()
        
        print('==> finised LayerQuantizier simulation. Appropriate option NOT found!')
        return None
                    
    def save_state(self):
        save_to_file((self.resume_index, self.is_final), False, cfg.RESULTS_DIR, self.resume_param_filename)
        
    def _update_layer(self, l_idx, p_idx):
        opt = self.input[l_idx][p_idx]
        if opt[0] == -1:
            self.sp_list[l_idx] = torch.from_numpy(mf.tile_opt(self.layers_layout[l_idx], self.default_in_pattern))
        elif self.mode == Mode.UNIFORM_LAYER:
            self.sp_list[l_idx] = torch.from_numpy(mf.tile_opt(self.layers_layout[l_idx], self.input_patterns[:,:,opt[0]]))
        elif self.mode == Mode.UNIFORM_FILTERS:
            self.sp_list[l_idx] = torch.from_numpy(mf.tile_opt(self.layers_layout[l_idx],self.input_patterns[l_idx][0][opt[0]]))
        else: 
            self.sp_list[l_idx] = torch.from_numpy(self.input_patterns[l_idx][opt[0]])
    
    def _save_final_rec(self,test_acc,ops_saved, ops_total):
        f_rec = FinalResultRc(test_acc,ops_saved, ops_total, self.mode, self.sp_list, \
                                 cfg.PS,cfg.MAX_ACC_LOSS,cfg.ONES_RANGE, cfg.NET.__name__)
        save_to_file(f_rec,True,cfg.RESULTS_DIR)
        print('==> finished LayerQuantizier simulation!')
        print('==> result saved to ' + f_rec.filename)
        self.is_final = [-2]*len(self.is_final)
        self.save_state()
        return f_rec
        
    def _is_finised(self):
        return self.is_final==[-2]*len(self.is_final)
        
    def _get_next_opt(self):
        next_idx = self.is_final.copy()
        for l,i in enumerate(self.resume_index):
            if i+1 < (len(self.input[l])-1):
                next_idx[l] = i+1
        for l,i in enumerate(next_idx):
            if next_idx[l] >= 0:
                ops_diff = self.input[l][i][1]-self.input[l][i-1][1]
                acc_diff = self.input[l][i-1][2]-self.input[l][i][2]
                next_idx[l] = ops_diff*acc_diff
        l_to_inc = next_idx.index(max(next_idx))
        if self.is_final[l_to_inc] == -2:
            return None
        self.resume_index[l_to_inc] = self.resume_index[l_to_inc]+1
        if self.resume_index[l_to_inc] == (len(self.input[l_to_inc])-1):
            self.is_final[l_to_inc] = -2
        return l_to_inc
        
    def _clean_input(self):
        for l in range(len(self.input)):
            self.curr_acc_th = 0
            self.input[l][:] = [tup for tup in self.input[l] if self._determine(tup)]
        del self.curr_acc_th
    
    def _determine(self, tup):
        p_idx, ops_saved, acc = tup
        if ops_saved == 0:
            return False
        elif self.curr_acc_th >= acc and acc > 0:
            return False
        else:
            self.curr_acc_th = acc
        return True
    
#    def simulatev2(self):
#       
#        print('==> starting simulation.')
#        
#        nn = net.NeuralNet()
#        test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
#        _, test_acc, _ = nn.test(test_gen)
#        print(f'==> Asserted test-acc of: {test_acc}\n')
#        
#        nn.net.initialize_spatial_layers(CIFAR10_shape(), cfg.BATCH_SIZE, cfg.PS)
#        _, test_acc, _ = nn.test(test_gen)
#        print(f'==> Asserted test-acc of: {test_acc}\n')
#        
#        all_net_opts = product(*self.input)
#        net_opts = islice(all_net_opts, self.resume_index,None)
#        
#        for i, net_opt in enumerate(net_opts,start=self.resume_index):
#            sp_list = [None]*len(self.input)
#            for idx, opt in enumerate(net_opt):
#                if opt[0] == -1:
#                    sp_list[idx] = torch.from_numpy(mf.tile_opt(self.layers_layout[idx], self.default_in_pattern))
#                elif self.mode == rc.uniform_layer:
#                    sp_list[idx] = torch.from_numpy(mf.tile_opt(self.layers_layout[idx], self.input_patterns[:,:,opt[0]]))
#                elif self.mode == rc.uniform_filters:
#                    sp_list[idx] = torch.from_numpy(mf.tile_opt(self.layers_layout[idx],self.input_patterns[idx][0][opt[0]]))
#                else: 
#                    sp_list[idx] = torch.from_numpy(self.input_patterns[idx][opt[0]])
#                
#            nn.net.strict_mask_update(update_ids=list(range(len(self.layers_layout))), masks=sp_list)
#            _, test_acc, _ = nn.test(test_gen)
#            ops_saved, ops_total = nn.net.num_ops()
#            if test_acc >= self.min_acc:
#                f_rec = rc.FinalResultRc(test_acc,ops_saved, ops_total, self.mode, sp_list, \
#                                         cfg.PS,cfg.MAX_ACC_LOSS,cfg.ONES_RANGE, cfg.NET.__name__)
#                rc.save_to_file(f_rec,True,cfg.RESULTS_DIR)
#                print('==> finished LayerQuantizier simulation!')
#                print('==> result saved to ' + f_rec.filename)
#                break
#            self.resume_index = i
#            self.save_state()
        
#test
#in_rec = rc.load_from_file('ps2_ones1x3_uniform_layer_acc93.83_mg1024_1543955881.pkl',cfg.RESULTS_DIR)
#lQ = LayerQuantizier(in_rec,90, 2)
#lQ.simulate()
