# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:22:47 2018

@author: Inna
"""

import numpy as np
import torch
from tqdm import tqdm
from itertools import zip_longest

from Record import Mode, RecordType, Record, save_to_file
import Config as cfg
import maskfactory as mf


class ChannelQuantizier():
    def __init__(self, rec, init_acc, max_acc_loss, patch_size, out_rec=None, default_in_pattern=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.input = rec.gen_pattern_lists(init_acc - max_acc_loss)

        if default_in_pattern is not None:
            self.default_in_pattern = default_in_pattern
        elif rec.mode == Mode.MAX_GRANULARITY:
            self.default_in_pattern = np.ones((1, 1), dtype=self.input_patterns[0][0][0].dtype)
        else:
            self.default_in_pattern = np.ones((self.input_patterns.shape[0], self.input_patterns.shape[0]),
                                              dtype=self.input_patterns.dtype)

        if out_rec is None:
            self._generate_patterns(rec.mode, rec.layers_layout, rec.gran_thresh, rec.filename, max_acc_loss, init_acc)
        else:
            self.output_rec = out_rec

    def simulate(self, nn, test_gen):
        st_point = self.output_rec.find_resume_point()
        print('==> starting ChannelQuantizier simulation.')

        save_counter = 0
        for layer in tqdm(range(st_point[0], len(self.input))):
            for p_idx in range(st_point[3], len(self.output_rec.all_patterns[layer])):
                st_point[3] = 0
                nn.net.strict_mask_update(update_ids=[layer],
                                          masks=[torch.from_numpy(self.output_rec.all_patterns[layer][p_idx])])
                _, test_acc, _ = nn.test(test_gen)
                ops_saved, ops_total = nn.net.num_ops()
                nn.net.reset_ops()
                self.output_rec.addRecord(ops_saved, ops_total, test_acc, layer, 0, 0, p_idx)

                save_counter += 1
                if save_counter > cfg.SAVE_INTERVAL:
                    self.save_state()
                    save_counter = 0

        self.save_state()
        print('==> finised ChannelQuantizier simulation.')

    def is_finised(self):
        return self.output_rec.find_resume_point() is None

    def save_state(self):
        save_to_file(self.output_rec, True, cfg.RESULTS_DIR)

    def _generate_patterns(self, mode, layers_layout, gran_thresh, rec_in_filename, max_acc_loss, init_acc):
        # TODO - fix Record init
        self.output_rec = Record(layers_layout, gran_thresh, False, mode, init_acc)
        self.output_rec.set_results_dimensions(no_of_layers=len(self.input),
                                               no_of_channels=[1] * len(self.input),
                                               no_of_patches=[1] * len(self.input))

        no_of_patterns_gen, all_patterns = self._gen_patterns_zip_longest(mode, layers_layout)
        self.output_rec.set_all_patterns(all_patterns, RecordType.cQ_REC)
        self.output_rec.set_results_dimensions(no_of_patterns=no_of_patterns_gen)

        self.output_rec.set_filename('ChannelQ_ma' + str(max_acc_loss) + '_' + rec_in_filename)

    def _gen_patterns_zip_longest(self, mode, layers_layout):
        all_patterns = []
        no_of_patterns = 0
        input_new = []
        no_of_patterns = [None] * len(self.input)
        for l in range(len(self.input)):
            layers = []
            input_new.append([self.input[l][c][0] for c in range(layers_layout[l][0])])
            for layer_opt in zip_longest(*input_new[l], fillvalue=(-1, -1, -1)):
                layer = np.ones(layers_layout[l], dtype=self.default_in_pattern.dtype)
                for idx, opt in enumerate(layer_opt):
                    if opt[0] == -1:
                        layer[idx, :, :] = mf.tile_opt((layers_layout[l][1], layers_layout[l][2]),
                                                       self.default_in_pattern, False)
                    elif mode == Mode.MAX_GRANULARITY:
                        layer[idx, :, :] = mf.tile_opt((layers_layout[l][1], layers_layout[l][2]),
                                                       self.input_patterns[l][idx][opt[0]], False)
                    else:
                        layer[idx, :, :] = mf.tile_opt((layers_layout[l][1], layers_layout[l][2]),
                                                       self.input_patterns[:, :, opt[0]], False)
                layers.append(layer)
            no_of_patterns[l] = len(layers)
            all_patterns.append(layers)
        self.input = input_new
        return no_of_patterns, all_patterns

# Test
# RESULTS_DIR = './data/results/'
# uniform_patch_res = 'ps2_ones(1, 3)_uniform_patch_acc93.83_mg1024_17860D15h.pkl'
# rec_in = rc.load_from_file(uniform_patch_res,RESULTS_DIR)
# cQ = ChannelQuantizier(rec_in,92,2)
