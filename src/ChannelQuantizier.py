# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 21:22:47 2018

@author: Inna
"""

import numpy as np
import torch
from tqdm import tqdm
from itertools import zip_longest
from operator import itemgetter
import random
import math

from Record import Mode, RecordType, Record, save_to_file
import Config as cfg
import maskfactory as mf

CQ_DEBUG = False

class ChannelQuantizier():
    def __init__(self, rec, init_acc, max_acc_loss, patch_size, out_rec=None, default_in_pattern=None):
        self.patch_size = patch_size
        self.input_patterns = rec.all_patterns
        self.input = rec.gen_pattern_lists(init_acc - max_acc_loss)
        
        input_new = []
        for l in range(len(self.input)):
            input_new.append([self.input[l][c][0] for c in range(rec.layers_layout[l][0])])
        self.input = input_new
        if cfg.CQ_OPTION == 2:
            self._clean_input()

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
            
    def number_of_iters(self):
        return sum(self.output_rec.no_of_patterns)

    def simulate(self, nn, test_gen):
        st_point = self.output_rec.find_resume_point()
        print('==> starting ChannelQuantizier simulation.')

        save_counter = 0
        for layer in tqdm(range(st_point[0], len(self.input))):
            for p_idx in range(st_point[3], len(self.output_rec.all_patterns[layer])):
                st_point[3] = 0
                nn.net.strict_mask_update(update_ids=[layer],
                                              masks=[torch.from_numpy(self.output_rec.all_patterns[layer][p_idx])])
                if CQ_DEBUG:
                    test_acc = 100
                    ops_saved = 100
                    ops_total = 100
                else:
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
        self.output_rec = Record(layers_layout, gran_thresh, False, mode, init_acc)
        self.output_rec.set_results_dimensions(no_of_layers=len(self.input),
                                               no_of_channels=[1] * len(self.input),
                                               no_of_patches=[1] * len(self.input))
        if cfg.CHANNELQ_UPDATE_RATIO == 1:
            no_of_patterns_gen, all_patterns = self._gen_patterns_zip_longest(mode, layers_layout)
        else:
            no_of_patterns_gen, all_patterns = self._gen_patterns_zip_ratio(mode, layers_layout)
        
        self.output_rec.set_all_patterns(all_patterns, RecordType.cQ_REC)
        self.output_rec.set_results_dimensions(no_of_patterns=no_of_patterns_gen)

        fn = f'ChannelQ{cfg.CQ_OPTION}r{cfg.CHANNELQ_UPDATE_RATIO}_ma{max_acc_loss}_' + rec_in_filename
        if CQ_DEBUG:
            fn = 'DEBUG_' + fn
        self.output_rec.set_filename(fn)
        
    def _build_layer(self, layer_dims, layer_opt, l, mode):
        layer = np.ones(layer_dims, dtype=self.default_in_pattern.dtype)
        for idx, opt in enumerate(layer_opt):
            if type(opt) is int:
                opt = self.input[l][idx][opt]
            if opt[0] == -1:
                tmp = mf.tile_opt((layer_dims[1], layer_dims[2]),
                                               self.default_in_pattern, False)
            elif mode == Mode.MAX_GRANULARITY:
                tmp = mf.tile_opt((layer_dims[1], layer_dims[2]),
                                               self.input_patterns[l][idx][opt[0]], False)
            else:
                tmp = mf.tile_opt((layer_dims[1], layer_dims[2]),
                                               self.input_patterns[:, :, opt[0]], False)
            layer[idx, :, :] = tmp[0:layer_dims[1], 0:layer_dims[2]]
        return layer
    
    def _gen_patterns_zip_ratio(self, mode, layers_layout):
        all_patterns = []
        no_of_patterns = [None] * len(self.input)
        for l in range(len(self.input)):
            channels_to_update = math.ceil(cfg.CHANNELQ_UPDATE_RATIO*layers_layout[l][0])
            layers = []
            curr_layer_opt = [0]*layers_layout[l][0]
            while curr_layer_opt is not None:
                layers.append(self._build_layer(layers_layout[l], curr_layer_opt, l, mode))
                curr_layer_opt = self._zip_ratio(curr_layer_opt, channels_to_update, self.input[l])
            no_of_patterns[l] = len(layers)
            all_patterns.append(layers)
        return no_of_patterns, all_patterns
    
    def _zip_ratio(self, curr_layer_opt, channels_to_update, layer_input):
        possible_channels = []
        last_stage_channels = []
        for channel, opt_idx in enumerate(curr_layer_opt):
            if cfg.TWO_STAGE and opt_idx+1 < (len(layer_input[channel])-1):
                possible_channels.append(channel)
            elif cfg.TWO_STAGE and opt_idx+1 == (len(layer_input[channel])-1):
                last_stage_channels.append(channel)
            elif not cfg.TWO_STAGE and opt_idx+1 < (len(layer_input[channel])):
                possible_channels.append(channel)
        
        if channels_to_update < len(possible_channels):
            if cfg.CQ_OPTION == 2:
                channels_to_inc = random.sample(possible_channels, channels_to_update)
            elif cfg.CQ_OPTION == 1:
                sorted_channels = [x for x,_ in sorted(zip(possible_channels, itemgetter(*possible_channels)(curr_layer_opt)), key=lambda pair: pair[1])]
                if channels_to_update == 1:
                    channels_to_inc = [sorted_channels[0]]
                else:
                    channels_to_inc = sorted_channels[:channels_to_update]
        elif len(possible_channels) > 0:
            channels_to_inc = possible_channels
        else:
            channels_to_inc = []
            
        additional_channels = channels_to_update - len(channels_to_inc)
        if additional_channels > 0:
            if additional_channels < len(last_stage_channels):
                if cfg.CQ_OPTION == 2:
                    channels_to_inc += random.sample(last_stage_channels, additional_channels)
                elif cfg.CQ_OPTION == 1:
                    if additional_channels == 1:
                        channels_to_inc.append(last_stage_channels[0])
                    else:
                        channels_to_inc += last_stage_channels[:additional_channels]
            elif len(last_stage_channels) > 0:
                channels_to_inc += last_stage_channels
        
        if len(channels_to_inc) == 0:
            return None
        else:
            for channel in channels_to_inc:
                curr_layer_opt[channel] += 1
            return curr_layer_opt 

    def _gen_patterns_zip_longest(self, mode, layers_layout):
        all_patterns = []
        no_of_patterns = [None] * len(self.input)
        for l in range(len(self.input)):
            layers = []
            for layer_opt in zip_longest(*self.input[l], fillvalue=(-1, -1, -1)):
                layers.append(self._build_layer(layers_layout[l], layer_opt, l, mode))
            no_of_patterns[l] = len(layers)
            all_patterns.append(layers)
        return no_of_patterns, all_patterns
    
    def _clean_input(self):
        for l in range(len(self.input)):
            for c in range(len(self.input[l])):
                self.input[l][c][:] = [tup for tup in self.input[l][c] if self._determine(tup)]


    def _determine(self, tup):
        p_idx, ops_saved, acc, tot_ops = tup
        if ops_saved == 0 and p_idx != -1:
            return False
        return True
