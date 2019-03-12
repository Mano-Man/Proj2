# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:51:55 2018

@author: Inna
"""
import numpy as np
import torch

from Record import Mode, FinalResultRc, load_from_file, save_to_file
import Config as cfg
from Config import DATA as dat
import maskfactory as mf
from itertools import product
import csv
import os

LQ_DEBUG = False

class LayerQuantResumeRec():
    def __init__(self,no_of_layers, input_length, max_acc_loss, inp):
        self.resume_index = [0]*(no_of_layers)
        self.is_final = [False]*(no_of_layers)
        for layer, idx in enumerate(self.resume_index):
            if idx==(input_length[layer]-1):
                self.is_final[layer] = True
        self.curr_saved_ops = 0
        self.curr_tot_ops = 0
        self.curr_best_acc = 0
        self.curr_resume_index = self.resume_index.copy()
        self.curr_best_mask = None
        self.input = inp
        self.algo_debug = []
        
    def find_first_unfinished_layer(self, should_mark=False):
        for layer, is_last in reversed(list(enumerate(self.is_final))):
            if not is_last:
                if should_mark:
                    self.mark_finished(layer=layer)
                return layer
        return None
    
    def mark_finished(self, layer=None, mark_all=False):
        if mark_all:
            self.is_final = [True]*len(self.is_final)
        elif layer is not None:
            self.is_final[layer] = True

    def is_finised(self):
        return (self.is_final==[True]*len(self.is_final) or (self.resume_index is None))
        
    def add_algo_debug_rec(self, acc, ops_saved, tot_ops):
        self.algo_debug.append([self.resume_index.copy(), acc, ops_saved, tot_ops, ops_saved/tot_ops])
        
    def find_best_mask(self, min_acc):
        indexes = None
        best_ops_saved = 0
        best_acc = min_acc
        for opt, acc, ops_saved,_,_ in self.algo_debug:
            if acc >= min_acc and best_ops_saved < ops_saved:
                indexes = opt
                best_ops_saved = ops_saved
                best_acc = acc
        return indexes, best_ops_saved, best_acc
        
    def save_csv(self, fn):
        out_path = os.path.join(cfg.RESULTS_DIR, fn + ".csv")
        with open(out_path, 'w', newline='') as f:
            csv.writer(f).writerow([len(l)-1 for l in self.input])
            csv.writer(f).writerow(self.input)
            csv.writer(f).writerow(
                ['option', 'accuracy', 'operations saved', 'total operations', '% saved'])
            for r in self.algo_debug:
                csv.writer(f).writerow(r)
        


class LayerQuantizier():
    def __init__(self, rec, init_acc, max_acc_loss, patch_size, ones_range, total_ops, resume_param_path=None, default_in_pattern=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.min_acc = init_acc - max_acc_loss
        self.mode = rec.mode
        self.layers_layout = rec.layers_layout
        self.max_acc_loss = max_acc_loss
        self.ones_range = ones_range
        self.total_ops = total_ops
        
        self.input = rec.gen_pattern_lists(self.min_acc)
        self.input = [self.input[l][0][0] for l in range(len(self.input))]
        self.no_of_patterns = 1
        for l_idx in range(len(self.input)):
            if cfg.LQ_OPTION == cfg.LQ_modes.DEFAULT:
                self.no_of_patterns += len(self.input[l_idx])
            elif cfg.LQ_OPTION == cfg.LQ_modes.PRODUCT:
                self.no_of_patterns *= len(self.input[l_idx])
        self._clean_input()
        self.product_iter = None
        
        if default_in_pattern is not None:
            self.default_in_pattern = default_in_pattern
        elif rec.mode == Mode.UNIFORM_LAYER:
            self.default_in_pattern = np.ones((self.input_patterns.shape[0], self.input_patterns.shape[0]),
                                              dtype=self.input_patterns.dtype)
        elif rec.mode == Mode.UNIFORM_FILTERS:
            self.default_in_pattern = np.ones((self.patch_size, self.patch_size), dtype=self.input_patterns[0][0][0].dtype)
        else:
            self.default_in_pattern = np.ones((self.patch_size, self.patch_size), dtype=self.input_patterns[0][0].dtype)

        self.resume_param_filename = f'LayerQ{cfg.LQ_OPTION.value}_ma' + str(max_acc_loss) + '_' + rec.filename
        if LQ_DEBUG:
            self.resume_param_filename = 'DEBUG_' + self.resume_param_filename

        if resume_param_path is None:
            input_length = [len(self.input[layer]) for layer in range(len(self.input))]
            self.resume_rec = LayerQuantResumeRec(len(self.input), input_length, max_acc_loss, self.input)
        else:
            self.resume_rec = load_from_file(resume_param_path, path='')
        
    def number_of_iters(self):
        return self.no_of_patterns


    def simulate(self, nn, test_gen):
        print('==> starting LayerQuantizier simulation.')

        self.sp_list = [None] * len(self.input)
        for l_idx, p_idx in enumerate(self.resume_rec.resume_index):
            self._update_layer( l_idx, p_idx)
                
        nn.net.strict_mask_update(update_ids=list(range(len(self.layers_layout))), masks=self.sp_list)
        if LQ_DEBUG:
            test_acc = 100
            ops_saved = 100
            ops_total = 100
        else:
            _, test_acc, _ = nn.test(test_gen)
            ops_saved, ops_total = nn.net.num_ops()
            nn.net.reset_ops()
        self.save_state(test_acc, ops_saved, ops_total)

        save_counter = 0 
        l_to_inc = self._get_next_opt()
        while (l_to_inc is not None):
            if cfg.LQ_OPTION == cfg.LQ_modes.PRODUCT:
                for l_idx, p_idx in enumerate(self.resume_rec.resume_index):
                    self._update_layer( l_idx, p_idx)
                nn.net.strict_mask_update(update_ids=list(range(len(self.layers_layout))), masks=self.sp_list)
            else:
                self._update_layer(l_to_inc, self.resume_rec.resume_index[l_to_inc])
                nn.net.lazy_mask_update(update_ids=[l_to_inc], masks=[self.sp_list[l_to_inc]])
                
            if LQ_DEBUG:
                test_acc = 100
                ops_saved = 100
                ops_total = 100
            else:
                _, test_acc, _ = nn.test(test_gen)
                ops_saved, ops_total = nn.net.num_ops()
                nn.net.reset_ops()
                
            save_counter += 1
            if save_counter > cfg.SAVE_INTERVAL:
                self.save_state(test_acc, ops_saved, ops_total, True)
                save_counter = 0
            else:
                self.save_state(test_acc, ops_saved, ops_total)
            l_to_inc = self._get_next_opt()

        self._save_final_rec()
        print(f'==> finised LayerQuantizier simulation')

    def is_finised(self):
        return self.resume_rec.is_finised()

    def save_state(self, test_acc, ops_saved, ops_total, to_file=False):
        self.resume_rec.add_algo_debug_rec(test_acc, ops_saved, ops_total)
        if test_acc >=  self.min_acc and ops_saved > self.resume_rec.curr_saved_ops:
            self.resume_rec.curr_best_acc = test_acc
            self.resume_rec.curr_saved_ops = ops_saved
            self.resume_rec.curr_tot_ops = ops_total
            self.resume_rec.curr_best_mask = [self.sp_list[l].clone() for l in range(len(self.sp_list))]
            self.resume_rec.curr_resume_index = self.resume_rec.resume_index.copy()
        if to_file:
            save_to_file(self.resume_rec, False, cfg.RESULTS_DIR, self.resume_param_filename)
            print(self.resume_param_filename)

    def max_number_of_iters(self, mock_input):
        input_length = [len(mock_input[layer]) for layer in range(len(mock_input))]
        return sum(input_length)

    def _update_layer(self, l_idx, p_idx):
        opt = self.input[l_idx][p_idx]
        if opt[0] == -1:
            self.sp_list[l_idx] = torch.from_numpy(mf.tile_opt(self.layers_layout[l_idx], self.default_in_pattern))
        elif self.mode == Mode.UNIFORM_LAYER:
            self.sp_list[l_idx] = torch.from_numpy(
                mf.tile_opt(self.layers_layout[l_idx], self.input_patterns[:, :, opt[0]]))
        elif self.mode == Mode.UNIFORM_FILTERS:
            self.sp_list[l_idx] = torch.from_numpy(
                mf.tile_opt(self.layers_layout[l_idx], self.input_patterns[l_idx][0][opt[0]]))
        else:
            self.sp_list[l_idx] = torch.from_numpy(self.input_patterns[l_idx][opt[0]])


    def _save_final_rec(self):
        self.resume_rec.save_csv(self.resume_param_filename)
        save_to_file(self.resume_rec, False, cfg.RESULTS_DIR, self.resume_param_filename)
        if self.resume_rec.curr_tot_ops == 0:
            print(f'==> No suitable Option was found for min accuracy of {self.min_acc}')
            return
        f_rec = FinalResultRc(self.min_acc + self.max_acc_loss, self.resume_rec.curr_best_acc, self.resume_rec.curr_saved_ops, 
                              self.resume_rec.curr_tot_ops, self.mode, self.resume_rec.curr_best_mask, 
                              self.patch_size, self.max_acc_loss, self.ones_range, cfg.NET.__name__, 
                              dat.name(), self.layers_layout)
        save_to_file(f_rec,True,cfg.RESULTS_DIR)
        print('==> result saved to ' + f_rec.filename)
        self.resume_rec.mark_finished(mark_all=True)
        save_to_file(self.resume_rec, False, cfg.RESULTS_DIR, self.resume_param_filename)
        return f_rec

    def _get_next_opt(self, nn=None, test_gen=None, curr_acc=None):
        if cfg.LQ_OPTION == cfg.LQ_modes.PRODUCT:
            if self.product_iter is None:
                lengths = [range(len(l)) for l in self.input]
                self.product_iter = product(*lengths)
                p = list(next(self.product_iter, None))
                while p != self.resume_rec.resume_index:
                    p = list(next(self.product_iter, None))
            p = next(self.product_iter, None)
            if p is not None:
                self.resume_rec.resume_index = list(p)
                return self.resume_rec.resume_index
            else:
                self.resume_rec.resume_index = None
                return None
            
        else:
            possible_layers = []
            current_sum_ops_saved = self._current_sum_ops_saved()
            for layer,opt_idx in enumerate(self.resume_rec.resume_index):
                if cfg.TWO_STAGE and opt_idx+1 < (len(self.input[layer])-1):
                    possible_layers.append(layer)
                elif not cfg.TWO_STAGE and opt_idx+1 < len(self.input[layer]):
                    possible_layers.append(layer)
            if len(possible_layers) > 0 :
                test_arr = [None]*len(possible_layers)
                for idx, layer in enumerate(possible_layers):
                    i = self.resume_rec.resume_index[layer]
                    acc_diff = self.input[layer][i+1][2]-self.input[layer][i][2]
                    if cfg.LQ_OPTION == cfg.LQ_modes.DEFAULT:
                        ops_diff = (current_sum_ops_saved - self._next_sum_ops_saved(layer,current_sum_ops_saved) + 1)/(current_sum_ops_saved + 1)
                    test_arr[idx] = acc_diff/ops_diff
    
                idx_to_inc = max(reversed(range(len(test_arr))), key=test_arr.__getitem__)
                l_to_inc = possible_layers[idx_to_inc]
                
            elif cfg.TWO_STAGE:
                l_to_inc = self.resume_rec.find_first_unfinished_layer(should_mark=True)
            else:
                l_to_inc = None
            
            if l_to_inc is not None:
                self.resume_rec.resume_index[l_to_inc] = self.resume_rec.resume_index[l_to_inc]+1
                if cfg.TWO_STAGE and (len(self.input[layer])-1)==self.resume_rec.resume_index[l_to_inc]:
                    self.resume_rec.mark_finished(layer=l_to_inc)
    
            return l_to_inc
    
    def _next_sum_ops_saved(self, l, curr_sum):
        sum_ops_saved = curr_sum - self.input[l][self.resume_rec.resume_index[l]][1]
        sum_ops_saved += self.input[l][self.resume_rec.resume_index[l]+1][1]
        return sum_ops_saved
    
    def _current_sum_ops_saved(self):
        sum_ops_saved = 0 
        for l, i in enumerate(self.resume_rec.resume_index):
            sum_ops_saved += self.input[l][i][1]
        return sum_ops_saved

    def _clean_input(self):
        for l in range(len(self.input)):
            self.curr_acc_th = 0
            self.input[l][:] = [tup for tup in self.input[l] if self._determine(tup)]
        del self.curr_acc_th

    def _determine(self, tup):
        p_idx, ops_saved, acc, tot_ops = tup
        if ops_saved == 0 and p_idx != -1:
            return False
        if cfg.LQ_OPTION == cfg.LQ_modes.CLEAN_DECREASING_ACC:
            if self.curr_acc_th >= acc and acc > 0:
                return False
            else:
                self.curr_acc_th = acc
        return True
    
    def find_final_mask(self, max_acc_loss, nn=None, test_gen=None, should_save=False):
        init_acc = self.max_acc_loss + self.min_acc
        new_min_acc = init_acc - max_acc_loss
        if not self.resume_rec.is_finised():
            print('Simulation not finished!')
            if nn is not None and test_gen is not None:
                self.simulate(nn, test_gen)
        final_mask_indexes, best_ops_saved, best_acc = self.resume_rec.find_best_mask(new_min_acc)
        if final_mask_indexes is None:
            print(f'==> No suitable Option was found for min accuracy of {new_min_acc}')
            return None
        self.sp_list = [None] * len(final_mask_indexes)
        for l_idx, p_idx in enumerate(final_mask_indexes):
            self._update_layer( l_idx, p_idx)
        f_rec = FinalResultRc(init_acc, best_acc, best_ops_saved, 
                              self.resume_rec.curr_tot_ops, self.mode, self.sp_list, 
                              self.patch_size, max_acc_loss, self.ones_range, cfg.NET.__name__, 
                              dat.name(), self.layers_layout)
        if should_save:
            save_to_file(f_rec,True,cfg.RESULTS_DIR)
            print('==> result saved to ' + f_rec.filename)
        return f_rec
            
        