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
    def __init__(self, no_of_layers, input_length, max_acc_loss):
        self.resume_index = [0] * (no_of_layers)
        self.final_placeholder = -round(max_acc_loss + 1) - 2
        self.is_final = [self.final_placeholder + 1] * (no_of_layers)
        for layer, idx in enumerate(self.resume_index):
            if idx == (input_length[layer] - 1):
                self.is_final[layer] = self.final_placeholder
        self.curr_saved_ops = 0
        self.curr_tot_ops = 0
        self.curr_best_acc = 0
        self.curr_best_mask = None
        self.reset_occured = False

    def reset(self, clean_input_length):
        no_of_layers = len(self.resume_index)
        self.resume_index = [0] * (no_of_layers)
        self.is_final = [self.final_placeholder + 1] * (no_of_layers)
        for layer, idx in enumerate(self.resume_index):
            if idx == (clean_input_length[layer] - 1):
                self.is_final[layer] = self.final_placeholder
        self.reset_occured = True

    def should_reset(self):
        return self.is_final == [self.final_placeholder] * len(self.is_final)

    def is_finised(self):
        return self.should_reset() and self.reset_occured

    def mock_reset(self):
        self.reset_occured = True


class LayerQuantizier():
    def __init__(self, rec, init_acc, max_acc_loss, patch_size, ones_range, resume_param_path=None,
                 default_in_pattern=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.min_acc = init_acc - max_acc_loss
        self.mode = rec.mode
        self.layers_layout = rec.layers_layout
        self.max_acc_loss = max_acc_loss
        self.ones_range = ones_range

        self.input = rec.gen_pattern_lists(self.min_acc)
        self.input = [self.input[l][0][0] for l in range(len(self.input))]
        # self._clean_input()

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
            self.resume_rec = LayerQuantResumeRec(len(self.input), input_length, max_acc_loss)
        else:
            self.resume_rec = load_from_file(resume_param_path, path='')
        self.resume_rec.mock_reset()

    def simulate(self, nn, test_gen):
        print('==> starting LayerQuantizier simulation.')

        self.sp_list = [None] * len(self.input)
        for l_idx, p_idx in enumerate(self.resume_rec.resume_index):
            # print(self.input)
            # print(f'{l_idx}, {p_idx}')
            self._update_layer(l_idx, p_idx)

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
        if test_acc >= self.min_acc and ops_saved > self.resume_rec.curr_saved_ops:
            self.resume_rec.curr_best_acc = test_acc
            self.resume_rec.curr_saved_ops = ops_saved
            self.resume_rec.curr_tot_ops = ops_total
            self.resume_rec.curr_best_mask = [self.sp_list[l].clone() for l in range(len(self.sp_list))]
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
        f_rec = FinalResultRc(self.min_acc + self.max_acc_loss, self.resume_rec.curr_best_acc,
                              self.resume_rec.curr_saved_ops,
                              self.resume_rec.curr_tot_ops, self.mode, self.resume_rec.curr_best_mask,
                              self.patch_size, self.max_acc_loss, self.ones_range, cfg.NET.__name__,
                              dat.name(), self.layers_layout)
        save_to_file(f_rec, True, cfg.RESULTS_DIR)
        # print('==> finished LayerQuantizier simulation!')
        print('==> result saved to ' + f_rec.filename)
        self.resume_rec.is_final = [self.resume_rec.final_placeholder] * len(self.resume_rec.is_final)
        save_to_file(self.resume_rec, False, cfg.RESULTS_DIR, self.resume_param_filename)
        return f_rec

    def _get_next_opt(self):
        next_idx = self.resume_rec.is_final.copy()
        for l, i in enumerate(self.resume_rec.resume_index):
            if i + 1 < (len(self.input[l]) - 1):
                next_idx[l] = i + 1
        for l, i in enumerate(next_idx):
            if next_idx[l] >= 0:
                ops_diff = self.input[l][i - 1][1] - self.input[l][i][1]
                acc_diff = self.input[l][i][2] - self.input[l][i - 1][2]
                next_idx[l] = (1 / (ops_diff + 1)) * acc_diff
        # l_to_inc = next_idx.index(max(next_idx))
        # print(next_idx)
        l_to_inc = max(reversed(range(len(next_idx))), key=next_idx.__getitem__)
        if self.resume_rec.is_final[l_to_inc] == self.resume_rec.final_placeholder:
            return None
        self.resume_rec.resume_index[l_to_inc] = self.resume_rec.resume_index[l_to_inc] + 1
        if self.resume_rec.resume_index[l_to_inc] == (len(self.input[l_to_inc]) - 1):
            self.resume_rec.is_final[l_to_inc] = self.resume_rec.final_placeholder
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
