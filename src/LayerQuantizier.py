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
        self.reset_occured = False
        self.input = inp
        self.test_acc_array = []
        self.ops_saved_array = []
        self.acc_diff_arr = []
        self.ops_diff_arr = []
        
    def find_first_unfinished_layer(self, should_mark=False):
        for layer, is_last in reversed(list(enumerate(self.is_final))):
            if not is_last:
                if should_mark:
                    self.mark_finished(layer=layer)
                return layer
        return None
        
    def reset(self, clean_input_length):
        no_of_layers = len(self.resume_index)
        self.resume_index = [0]*(no_of_layers)
        self.is_final = [False]*(no_of_layers)
        for layer, idx in enumerate(self.resume_index):
            if idx==(clean_input_length[layer]-1):
                self.is_final[layer] = True
        self.reset_occured = True

    def should_reset(self):
        return self.is_final==[True]*len(self.is_final)
    
    def mark_finished(self, layer=None, mark_all=False):
        if mark_all:
            self.is_final = [True]*len(self.is_final)
        elif layer is not None:
            self.is_final[layer] = True
    
    def is_finised(self):
        return self.should_reset() and self.reset_occured

    def mock_reset(self):
        self.reset_occured = True
        
    def add_rec(self, acc, ops_saved):
        self.test_acc_array.append(acc)
        self.ops_saved_array.append(ops_saved)
        
    def add_next_idx(self, acc_diff, ops_diff):
        self.acc_diff_arr.append(acc_diff)
        self.ops_diff_arr.append(ops_diff)
        


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
        #self._clean_input()
        assert len(self.input[0][0])==4
        print('option 4')
        
        if default_in_pattern is not None:
            self.default_in_pattern = default_in_pattern
        elif rec.mode == Mode.UNIFORM_LAYER:
            self.default_in_pattern = np.ones((self.input_patterns.shape[0], self.input_patterns.shape[0]),
                                              dtype=self.input_patterns.dtype)
        elif rec.mode == Mode.UNIFORM_FILTERS:
            self.default_in_pattern = np.ones((1, 1), dtype=self.input_patterns[0][0][0].dtype)
        else:
            self.default_in_pattern = np.ones((1, 1), dtype=self.input_patterns[0][0].dtype)

        self.resume_param_filename = 'LayerQ_ma' + str(max_acc_loss) + '_' + rec.filename

        if resume_param_path is None:
            input_length = [len(self.input[layer]) for layer in range(len(self.input))]
            self.resume_rec = LayerQuantResumeRec(len(self.input), input_length, max_acc_loss, self.input)
        else:
            self.resume_rec = load_from_file(resume_param_path, path='')
        self.resume_rec.mock_reset()
        
    def number_of_iters(self):
        no_of_patterns = [len(self.input[l_idx]) for l_idx in range(len(self.input))]
        return sum(no_of_patterns)

    def simulate(self, nn, test_gen):
        print('==> starting LayerQuantizier simulation.')

        self.sp_list = [None] * len(self.input)
        for l_idx, p_idx in enumerate(self.resume_rec.resume_index):
            self._update_layer( l_idx, p_idx)
                
        nn.net.strict_mask_update(update_ids=list(range(len(self.layers_layout))), masks=self.sp_list)
        _, test_acc, _ = nn.test(test_gen)
        ops_saved, ops_total = nn.net.num_ops()
        nn.net.reset_ops()
        self.save_state(test_acc, ops_saved, ops_total)

        counter = 1
        l_to_inc = self._get_next_opt()
        while (l_to_inc is not None):
            self._update_layer(l_to_inc, self.resume_rec.resume_index[l_to_inc])

            nn.net.lazy_mask_update(update_ids=[l_to_inc], masks=[self.sp_list[l_to_inc]])
            _, test_acc, _ = nn.test(test_gen)
            ops_saved, ops_total = nn.net.num_ops()
            nn.net.reset_ops()
            self.save_state(test_acc, ops_saved, ops_total)
            l_to_inc = self._get_next_opt()
            counter += 1

        self._save_final_rec()
        print(f'==> finised LayerQuantizier simulation')

    def is_finised(self):
        return self.resume_rec.is_finised()

    def save_state(self, test_acc, ops_saved, ops_total):
        self.resume_rec.add_rec(test_acc, ops_saved)
        if test_acc >=  self.min_acc and ops_saved > self.resume_rec.curr_saved_ops:
            self.resume_rec.curr_best_acc = test_acc
            self.resume_rec.curr_saved_ops = ops_saved
            self.resume_rec.curr_tot_ops = ops_total
            self.resume_rec.curr_best_mask = [self.sp_list[l].clone() for l in range(len(self.sp_list))]
            self.resume_rec.curr_resume_index = self.resume_rec.resume_index.copy()
        save_to_file(self.resume_rec, False, cfg.RESULTS_DIR, self.resume_param_filename)

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

    def _get_next_opt(self):
        possible_layers = []
        for layer,opt_idx in enumerate(self.resume_rec.resume_index):
            if opt_idx+1 < (len(self.input[layer])-1):
                possible_layers.append(layer)
        if len(possible_layers) > 0 :
            test_arr = [None]*len(possible_layers)
            for idx, layer in enumerate(possible_layers):
                i = self.resume_rec.resume_index[layer]
                #ops_diff = 10*100*(self.input[l][i-1][1]-self.input[l][i][1]+1)/self.total_ops 
                acc_diff = self.input[layer][i+1][2]-self.input[layer][i][2]
                ops_diff = 100*((self.input[layer][i][1]-self.input[layer][i+1][1]+1)/self.total_ops) 
                #ops_diff = self.input[layer][i][1]-self.input[layer][i+1][1]
                
                self.resume_rec.add_next_idx(acc_diff,ops_diff)

                test_arr[idx] = (1/(ops_diff))*acc_diff

            idx_to_inc = max(reversed(range(len(test_arr))), key=test_arr.__getitem__)
            l_to_inc = possible_layers[idx_to_inc]
        else:
            l_to_inc = self.resume_rec.find_first_unfinished_layer(should_mark=True)
        
        if l_to_inc is not None:
            self.resume_rec.resume_index[l_to_inc] = self.resume_rec.resume_index[l_to_inc]+1
        return l_to_inc

    def _clean_input(self):
        for l in range(len(self.input)):
            self.curr_acc_th = 0
            self.input[l][:] = [tup for tup in self.input[l] if self._determine(tup)]
        del self.curr_acc_th

    def _determine(self, tup):
        p_idx, ops_saved, acc = tup
        if ops_saved == 0 and p_idx != -1:
            return False
        elif self.curr_acc_th >= acc and acc > 0:
            return False
        else:
            self.curr_acc_th = acc
        return True
