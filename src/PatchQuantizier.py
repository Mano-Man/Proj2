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
from operator import itemgetter

from Record import Mode, Record, RecordType, save_to_file
import Config as cfg
import maskfactory as mf

PQ_DEBUG = False

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
                        mask[c, :, :] = self.output_rec.all_patterns[l][c][p_idx]

                    nn.net.strict_mask_update(update_ids=[l], masks=[torch.from_numpy(mask)])
                    
                    if PQ_DEBUG:
                        test_acc = 100
                        ops_saved = 100
                        ops_total = 100
                    else:
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
        self.output_rec = Record(layers_layout, gran_thresh, False, mode, init_acc)
        self.output_rec.set_results_dimensions(no_of_layers=len(self.input), no_of_patches=[1] * len(self.input))
        
        if cfg.PATCHQ_UPDATE_RATIO == 1:
            no_of_patterns_gen, all_patterns = self._gen_patterns_zip_longest(layers_layout)
        else:
            no_of_patterns_gen, all_patterns = self._gen_patterns_zip_ratio(layers_layout)
        
        self.output_rec.set_all_patterns(all_patterns, RecordType.pQ_REC)
        self.output_rec.set_results_dimensions(no_of_patterns=no_of_patterns_gen,
                                               no_of_channels=[len(c) for c in all_patterns])
        
        fn = f'PatchQ{cfg.PQ_OPTION.value}r{cfg.PATCHQ_UPDATE_RATIO}_ma{max_acc_loss}_' + rec_in_filename
        if PQ_DEBUG:
            fn = 'DEBUG_' + fn
        self.output_rec.set_filename(fn)
        
    def _build_channel(self, channel_opt, l, c, layer_dims):
        new_patch_n = math.ceil(layer_dims[1] / self.actual_patch_sizes[l])
        new_patch_m = math.ceil(layer_dims[2] / self.actual_patch_sizes[l])
        channel = np.ones(
            (new_patch_n * self.actual_patch_sizes[l], new_patch_m * self.actual_patch_sizes[l]),
            dtype=self.input_patterns.dtype)
        for patch_idx, opt in enumerate(channel_opt):
            if type(opt) is int:
                opt = self.input[l][c][patch_idx][opt]
            ii, jj = mf.get_patch_indexes(patch_idx, layer_dims[1], self.actual_patch_sizes[l])
            p = self.input_patterns[:, :, opt[0]]
            if (self.actual_patch_sizes[l] != self.patch_size):
                p = mf.tile_opt((self.actual_patch_sizes[l], self.actual_patch_sizes[l]), p, False)
            channel = mf.change_one_patch2d(channel, ii, jj, self.actual_patch_sizes[l], p)
        return mf.crop(layer_dims, channel, self.patch_size)
    
    def _zip_ratio(self, curr_channel_opt, patches_to_update, channel_input):
        possible_patches = []
        last_stage_patches = []
        for patch, opt_idx in enumerate(curr_channel_opt):
            if cfg.TWO_STAGE and opt_idx+1 < (len(channel_input[patch])-1):
                possible_patches.append(patch)
            elif cfg.TWO_STAGE and opt_idx+1 == (len(channel_input[patch])-1):
                last_stage_patches.append(patch)
            elif not cfg.TWO_STAGE and opt_idx+1 < (len(channel_input[patch])):
                possible_patches.append(patch)
        
        if patches_to_update < len(possible_patches):
            if cfg.PQ_OPTION == cfg.PQ_modes.DEFAULT:
                sorted_patches = [x for x,_ in sorted(zip(possible_patches, itemgetter(*possible_patches)(curr_channel_opt)), key=lambda pair: pair[1])]
                if patches_to_update == 1:
                    patches_to_inc = [sorted_patches[0]]
                else:
                    patches_to_inc = list(itemgetter(*range(patches_to_update))(sorted_patches))
        elif len(possible_patches) > 0:
            patches_to_inc = possible_patches
        else:
            patches_to_inc = []
            
        additional_patches = patches_to_update - len(patches_to_inc)
        if additional_patches > 0:
            if additional_patches < len(last_stage_patches):
                if cfg.PQ_OPTION == cfg.PQ_modes.DEFAULT:
                    if additional_patches == 1:
                        patches_to_inc.append(last_stage_patches[0])
                    else:
                        patches_to_inc += list(itemgetter(*range(additional_patches))(last_stage_patches))
            elif len(last_stage_patches) > 0:
                patches_to_inc += last_stage_patches
        
        if len(patches_to_inc) == 0:
            return None
        else:
            for patch in patches_to_inc:
                curr_channel_opt[patch] += 1
            return curr_channel_opt
        
    def _gen_patterns_zip_ratio(self, layers_layout):
        all_patterns = []
        no_of_patterns = [None] * len(self.input)
        for l in range(len(self.input)):
            layers = []
            no_of_patterns[l] = [0] * len(self.input[l])
            for c in range(len(self.input[l])):
                channels = []
                patches_to_update = math.ceil(cfg.PATCHQ_UPDATE_RATIO*len(self.input[l][c]))
                curr_channel_opt = [0]*len(self.input[l][c])
                while curr_channel_opt is not None:
                    no_of_patterns[l][c] += 1
                    channels.append(self._build_channel(curr_channel_opt, l, c, layers_layout[l]))
                    curr_channel_opt = self._zip_ratio(curr_channel_opt, patches_to_update, self.input[l][c])
                layers.append(channels)
            all_patterns.append(layers)
        patterns_max_count = [max(no_of_patterns[l]) for l in range(len(self.input))]
        return patterns_max_count, all_patterns

    def _gen_patterns_zip_longest(self, layers_layout):
        all_patterns = []
        no_of_patterns = [None] * len(self.input)
        for l in range(len(self.input)):
            layers = []
            no_of_patterns[l] = [0] * len(self.input[l])
            for c in range(len(self.input[l])):
                channels = []
                for channel_opt in zip_longest(*self.input[l][c], fillvalue=(-1, -1, -1)):
                    no_of_patterns[l][c] += 1
                    channels.append(self._build_channel(channel_opt, l, c, layers_layout[l]))
                layers.append(channels)
            all_patterns.append(layers)
        patterns_max_count = [max(no_of_patterns[l]) for l in range(len(self.input))]
        return patterns_max_count, all_patterns

