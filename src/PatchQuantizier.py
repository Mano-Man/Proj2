# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:22:47 2018

@author: Inna
"""

import numpy as np
import torch
import math
from tqdm import tqdm
from itertools import zip_longest

from Record import Mode, Record, RecordType, save_to_file
import Config as cfg
import maskfactory as mf


class PatchQuantizier():
    def __init__(self, rec, init_acc, max_acc_loss, patch_size, out_rec=None, default_in_pattern=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.actual_patch_sizes = rec.patch_sizes
        self.input = rec.gen_pattern_lists(init_acc - max_acc_loss)

        if default_in_pattern is None:
            self.default_in_pattern = np.ones((self.input_patterns.shape[0], self.input_patterns.shape[0]),
                                              dtype=self.input_patterns.dtype)
        else:
            self.default_in_pattern = default_in_pattern

        if out_rec is None:
            self._generate_patterns(rec.mode, rec.layers_layout, rec.gran_thresh, rec.filename, max_acc_loss, init_acc)
        else:
            self.output_rec = out_rec
            
    def number_of_iters(self):
        return self.output_rec.size
            
    def simulate(self, nn, test_gen):       
        st_point = self.output_rec.find_resume_point()
        print('==> Starting PatchQuantizier simulation.')

        save_counter = 0
        for l in tqdm(range(st_point[0], len(self.input))):
            for c in range(st_point[1], len(self.output_rec.all_patterns[l])):
                st_point[1] = 0
                for p_idx in range(st_point[3], len(self.output_rec.all_patterns[l][c])):
                    st_point[3] = 0

                    if self.output_rec.mode == Mode.UNIFORM_FILTERS:
                        mask = mf.tile_opt(self.output_rec.layers_layout[l], self.output_rec.all_patterns[l][c][p_idx],
                                           True)
                    else:
                        mask = np.ones(self.output_rec.layers_layout[l], dtype=self.default_in_pattern.dtype)

                    nn.net.strict_mask_update(update_ids=[l], masks=[torch.from_numpy(mask)])
                    _, test_acc, _ = nn.test(test_gen)
                    ops_saved, ops_total = nn.net.num_ops()
                    nn.net.reset_ops()
                    self.output_rec.addRecord(ops_saved, ops_total, test_acc, l, c, 0, p_idx)

                    save_counter += 1
                    if save_counter > cfg.SAVE_INTERVAL:
                        self.save_state()
                        save_counter = 0

        self.output_rec.fill_empty()
        self.save_state()

        print('==> finised PatchQuantizier simulation.')

    def is_finised(self):
        return self.output_rec.find_resume_point() is None

    def save_state(self):
        save_to_file(self.output_rec, True, cfg.RESULTS_DIR)

    def _generate_patterns(self, mode, layers_layout, gran_thresh, rec_in_filename, max_acc_loss, init_acc):
        # TODO - fix Record init
        self.output_rec = Record(layers_layout, gran_thresh, False, mode, init_acc)
        self.output_rec.set_results_dimensions(no_of_layers=len(self.input), no_of_patches=[1] * len(self.input))

        no_of_patterns_gen, all_patterns = self._gen_patterns_zip_longest(layers_layout)
        self.output_rec.set_all_patterns(all_patterns, RecordType.pQ_REC)
        self.output_rec.set_results_dimensions(no_of_patterns=no_of_patterns_gen,
                                               no_of_channels=[len(c) for c in all_patterns])

        self.output_rec.set_filename('PatchQ_ma' + str(max_acc_loss) + '_' + rec_in_filename)

    def _gen_patterns_zip_longest(self, layers_layout):
        all_patterns = []
        no_of_patterns = [None] * len(self.input)
        for l in range(len(self.input)):
            layers = []
            no_of_patterns[l] = [0] * len(self.input[l])
            for c in range(len(self.input[l])):
                channels = []
                for channel_opt in zip_longest(*self.input[l][c], fillvalue=(-1, -1, -1)):
                    # channel = np.ones((layers_layout[l][1], layers_layout[l][2]), dtype=self.input_patterns.dtype)
                    new_patch_n = math.ceil(layers_layout[l][1] / self.actual_patch_sizes[l])
                    new_patch_m = math.ceil(layers_layout[l][2] / self.actual_patch_sizes[l])
                    channel = np.ones(
                        (new_patch_n * self.actual_patch_sizes[l], new_patch_m * self.actual_patch_sizes[l]),
                        dtype=self.input_patterns.dtype)
                    for patch_idx, opt in enumerate(channel_opt):
                        ii, jj = mf.get_patch_indexes(patch_idx, layers_layout[l][1], self.actual_patch_sizes[l])
                        p = self.input_patterns[:, :, opt[0]]
                        if (self.actual_patch_sizes[l] != self.patch_size):
                            p = mf.tile_opt((self.actual_patch_sizes[l], self.actual_patch_sizes[l]), p, False)
                        channel = mf.change_one_patch2d(channel, ii, jj, self.actual_patch_sizes[l], p)
                    no_of_patterns[l][c] += 1
                    if (self.actual_patch_sizes[l] != self.patch_size):
                        new_patch_n = math.ceil(layers_layout[l][1] / self.patch_size)
                        new_patch_m = math.ceil(layers_layout[l][2] / self.patch_size)
                    channels.append(channel[0:(new_patch_n * self.patch_size), 0:(new_patch_m * self.patch_size)])
                layers.append(channels)
            all_patterns.append(layers)
        patterns_max_count = [max(no_of_patterns[l]) for l in range(len(self.input))]
        return patterns_max_count, all_patterns

# Test
# RESULTS_DIR = './data/results/'
# uniform_patch_res = 'ps2_ones(1, 3)_uniform_patch_acc93.83_mg1024_17860D15h.pkl'
# rec_in = rc.load_from_file(uniform_patch_res,RESULTS_DIR)
# cQ = ChannelQuantizier(rec_in,92,2)
