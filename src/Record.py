# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
from bitarray import bitarray
from enum import Enum
import numpy as np
import pickle
import math
import time
import os
import csv

import Config as cfg

# ----------------------------------------------------------------------------------------------------------------------
#                                              Granularity Modes Definition
# ----------------------------------------------------------------------------------------------------------------------

class Mode(Enum):
    MAX_GRANULARITY = 0
    UNIFORM_FILTERS = 1
    UNIFORM_PATCH = 2
    UNIFORM_LAYER = 3
    
gran_dict = {Mode.MAX_GRANULARITY:"max_granularity", 
             Mode.UNIFORM_FILTERS:"uniform_filters", 
             Mode.UNIFORM_PATCH:"uniform_patch", 
             Mode.UNIFORM_LAYER:"uniform_layer"}

# ----------------------------------------------------------------------------------------------------------------------
#                                                    Util Functions
# ----------------------------------------------------------------------------------------------------------------------
def bitmasks(n, m):
    '''
    generates all possiable bitmasks for:
        n = length of bitmask
        m = number of '1's in the generated bitmasks
    '''
    if m < n:
        if m > 0:
            for x in bitmasks(n - 1, m - 1):
                yield bitarray([1]) + x
            for x in bitmasks(n - 1, m):
                yield bitarray([0]) + x
        else:
            yield n * bitarray('0')
    else:
        yield n * bitarray('1')


def patches(patch_size, ones_range, mask_type=np.float32):
    '''
    generates all possible patches of size [patch_size x patch_size] with number of '1's in ones_range
    ones_range is exclusive, i.e if ones_range==(1,3), patchs with one and two '1's will be generated,
    while patches with 3 '1's will NOT.
    '''
    n = patch_size * patch_size
    for m in range(*ones_range):
        for patch in bitmasks(n, m):
            yield np.array(patch, dtype=mask_type).reshape((patch_size, patch_size))


def all_patches_array(patch_size, ones_range, mask_type=np.float32):
    '''
    returns an array that contains all possible patches of size [patch_size x patch_size] 
    with number of '1's in ones_range.
    
    ones_range is an exclusive tuple:
        for example ones_range=(1,3) means patchs with one and two '1's will be generated,
        while patches with 3 '1's will NOT.
    
    usage:
        patterns = all_patches_array(patch_size, ones_range)
        number_of_patterns = patterns.shape[2]
        pattern_number_3 = patterns[:,:,3]
        
    '''
    all_patches = np.zeros((patch_size, patch_size, 1), dtype=mask_type)
    for p in patches(patch_size, ones_range, mask_type):
        all_patches = np.append(all_patches, p[:, :, np.newaxis], axis=2)
    return all_patches[:, :, 1:]


def actual_patch_size(N, M, patch_size, gran_thresh):
    '''
    returns new patch size, such that does not exceeds granularity threshold
    '''
    granularity = (N * M) / (patch_size * patch_size)
    new_patch_size = patch_size
    while granularity > gran_thresh:
        new_patch_size += patch_size
        granularity = (N * M) / (new_patch_size * new_patch_size)
    return new_patch_size

def save_to_file(record, use_default=True, path='./data/results', filename=''):
    '''
    saving record to file using pickle library
    if use_default==True, record should have a valid record.filename field
    '''
    if use_default:
        filename = record.filename
    if not os.path.isdir(path):
        os.mkdir(path)
    outfile = open(os.path.join(path, filename + '.pkl'), 'wb')
    pickle.dump(record, outfile)
    outfile.close()


def load_from_file(filename, path='./data/results'):
    '''
    loads object from file using pickle library and returns the object
    '''
    infile = open(os.path.join(path, filename), 'rb')
    record = pickle.load(infile)
    infile.close()
    return record

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Record Class
# ----------------------------------------------------------------------------------------------------------------------
class Record():
    '''
    Class for storing simulation results. 
    
    Initializing: 
        layers_layout - tuple of layer dimension tuples 
            ((channels_1st_layer,rows_1st_layer,colums_1st_layer),
             (channels_2nd_layer,rows_2nd_layer,colums_2nd_layer),... )  
        gran_thresh -   relevant only in max_granularity mode.
            if (rows_in_layer x colums_in_layer)/(patch_size x patch_size) > gran_thresh
            patch_size is increased
        gen_patches -   boolean
            if True, all possiable patterns for patches are generated and stored at 
            Record.all_patterns:
                argv[0] should contain patch_size
                argv[1] should contain ones_range, see details in all_patches_array function doc
            if False,
                argv[0] should contain all the patterns relevant for this Record. 
                        argv[0] is stored at Record.all_patterns
                argv[1] should contain a list of the format:
                    [no_of_layers,no_of_channels,no_of_patches,no_of_patterns]
                    where no_of_layers is an integer 
                    no_of_channels, no_of_patches, no_of_patterns are lists of 
                    no_of_layers length
                IMPORTANT! when gen_patches==False, after init yot HAVE to 
                call _create_results()
        mode -          granularity mode
        initial_acc -   initial accuracy of the model 
    
    Initializing Examples:
        with gen_patches == True:
            layers_layout = ((64,32,32), (128,16,16), (256,8,8)) # this is a network with 3 predictions layers
                                                                 # (64,32,32) are the dimensions of the required mask in the 1st layer
            rec = Record(layers_layout, GRAN_THRESH, True, max_granularity, 93.83, PS, ONES_RANGE)
        with gen_patches == False:
            patterns = ... # Data structure containing the 
            rec = Record(layers_layout, cfg.GRAN_THRESH, False, max_granularity, 93.83, patterns, \
                         [3, [64,128,256], [256,64,16], None])
            rec.no_of_patterns = [10,10,10]
            rec._create_results()   # IMPORTANT! when gen_patches==False, after init and after the 
                                    # dimenstions (no_of_layers, no_of_channels, no_of_patches, no_of_patterns)
                                    # are set yot HAVE to call _create_results()
                                    
    Simulation results are stored in Record.results:
        Record.results[layer][channel][patch][pattern] = (ops_saved, total_ops, accuracy)
               
    '''
    def __init__(self, layers_layout, gran_thresh, gen_patches, mode, initial_acc, *argv):
        self.mode = mode
        self.gran_thresh = gran_thresh
        self.layers_layout = layers_layout
        
        if gen_patches:
            self.no_of_layers = len(layers_layout)
            self.no_of_channels = [l[0] for l in layers_layout]
            self.all_patterns = all_patches_array(argv[0], argv[1])  # [patch_size, ones_range]
            
            patch_size = self.all_patterns.shape[0]
            if mode==Mode.MAX_GRANULARITY:
                self.patch_sizes = [actual_patch_size(l[1], l[2], patch_size, gran_thresh) for l in layers_layout]
            else:
                self.patch_sizes = [patch_size] * self.no_of_layers
            self.no_of_patches = [math.ceil(layers_layout[idx][1] / self.patch_sizes[idx]) * \
                                  math.ceil(layers_layout[idx][2] / self.patch_sizes[idx]) for idx in
                                  range(self.no_of_layers)]
    
            if mode == Mode.UNIFORM_FILTERS:
                self.no_of_channels = [1] * self.no_of_layers
            elif mode == Mode.UNIFORM_PATCH:
                self.no_of_patches = [1] * self.no_of_layers
            elif mode == Mode.UNIFORM_LAYER:
                self.no_of_channels = [1] * self.no_of_layers
                self.no_of_patches = [1] * self.no_of_layers
            self.no_of_patterns = [self.all_patterns.shape[2]]*self.no_of_layers
            self._create_results()
			
            self.filename = (f'{cfg.NET.__name__}_{cfg.DATA_NAME}_acc{initial_acc}'
                             f'_{gran_dict[self.mode]}_ps{argv[0]}_ones{argv[1][0]}x{argv[1][1]}'
                             f'_mg{round(gran_thresh,0)}_{int(time.time())}')
        else:
            self.all_patterns = argv[0]
            self.no_of_layers,self.no_of_channels, self.no_of_patches, self.no_of_patterns = argv[1]
            self.filename = ''
			#self.patch_sizes
        

    def _create_results(self):
        self.results = []
        self.size = 0
        for l in range(self.no_of_layers):
            layer = []
            for k in range(self.no_of_channels[l]):
                channel = []
                for j in range(self.no_of_patches[l]):
                    patch = []
                    for i in range(self.no_of_patterns[l]):
                        patch.append(None)
                        self.size += 1
                    channel.append(patch)
                layer.append(channel)
            self.results.append(layer)
    
    def addRecord(self, op, tot_op, acc, layer, channel=0, patch_idx=0, pattern_idx=0):
        self.results[layer][channel][patch_idx][pattern_idx] = (op, tot_op, acc)

    def find_resume_point(self):
        '''
        returns list of format [layer,channel,patch,pattern] with the first 
        indices for which simulation results were nor recorded
        '''
        for layer in range(self.no_of_layers):
            for channel in range(self.no_of_channels[layer]):
                for patch_idx in range(self.no_of_patches[layer]):
                    for pattern_idx in range(self.no_of_patterns[layer]):
                        if self.results[layer][channel][patch_idx][pattern_idx] is None:
                            return [layer, channel, patch_idx, pattern_idx]
    def fill_empty(self):
        for layer in range(self.no_of_layers):
            for channel in range(self.no_of_channels[layer]):
                for patch_idx in range(self.no_of_patches[layer]):
                    for pattern_idx in range(self.no_of_patterns[layer]):
                        if self.results[layer][channel][patch_idx][pattern_idx] is None:
                            self.results[layer][channel][patch_idx][pattern_idx] = (-1,-1,-1)
                        
    def is_full(self):
        return None==self.find_resume_point()

    def save_to_csv(self, path='./data/results'):
        out_path = os.path.join(path, self.filename + ".csv")
        with open(out_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['layer', 'channel', 'patch number', 'pattern number', 'operations saved', 'total operations',
                 'accuracy'])
            for l in range(self.no_of_layers):
                for k in range(self.no_of_channels[l]):
                    for j in range(self.no_of_patches[l]):
                        for i in range(self.no_of_patterns[l]):
                            if self.results[l][k][j][i] is not None:
                                csv.writer(f).writerow([l, k, j, i, self.results[l][k][j][i][0], \
                                                        self.results[l][k][j][i][1], \
                                                        self.results[l][k][j][i][2]])
            
    def gen_pattern_lists(self, min_acc):
        '''
        
        '''
        slresults = []
        for l in range(self.no_of_layers):
            layer = []
            for k in range(self.no_of_channels[l]):
                channel = []
                for j in range(self.no_of_patches[l]):
                    patch = []
                    for p_idx, res_tuple in sorted(enumerate(self.results[l][k][j][:]),key=lambda x:(x[1][0],x[1][2]),  reverse=True):
                        if res_tuple[2] >= min_acc:
                            #patch.append((p_idx,res_tuple[0],res_tuple[2]))
                            patch.append((p_idx,res_tuple[0]/res_tuple[1],res_tuple[2]))
                    patch.append((-1,-1,-1))
                    channel.append(patch)
                layer.append(channel)
            slresults.append(layer)
        return slresults
    
    
            
#    def _indexed_according_to_mode(self,layer, channel, patch_idx, pattern_idx):
#        if self.mode == uniform_filters:
#            channel = 0
#        elif self.mode == uniform_patch:
#            patch_idx = 0
#        elif self.mode == uniform_layer:
#            channel = 0
#            patch_idx = 0
#        return layer, channel, patch_idx, pattern_idx

# ----------------------------------------------------------------------------------------------------------------------
#                                               Final Result Record Class
# ----------------------------------------------------------------------------------------------------------------------    
class FinalResultRc():
    def __init__(self, init_acc, f_acc, ops_saved, tot_ops, mode, pattern,ps,max_acc_loss, ones_range, net_name, dataset_name):
        self.filename = f'FR_{net_name}_ps{ps}_ones{ones_range[0]}x{ones_range[1]}_{gran_dict[mode]}_ma{max_acc_loss}_os{round((ops_saved/tot_ops)*100, 3)}_fa{f_acc}'
        self.mask = pattern
        self.final_acc = f_acc
        self.ops_saved = ops_saved
        self.total_ops = tot_ops
        self.mode = mode
        self.max_acc_loss = max_acc_loss
        self.patch_size = ps
        self.ones_range = ones_range
        self.network = net_name
        self.dataset_name = dataset_name
        self.init_acc = init_acc
        
    def __str__(self):
        string =  "================================================================\n"
        string += " RESULT FOR:    {:>15} {:>10} {:>20}\n".format(self.network, self.dataset_name, gran_dict[self.mode])
        string += "                {:>15} {}\n".format("TEST SET SIZE:", cfg.TEST_SET_SIZE)
        string += "                {:>15} {}\n".format("INITIAL ACC:", self.init_acc)
        string += "                {:>15} {}\n".format("PATCH SIZE:", self.patch_size)
        string += "                {:>15} {}-{}\n".format("ONES:", self.ones_range[0], self.ones_range[1])
        string += "                {:>15} {}\n".format("MAX ACC LOSS:", self.max_acc_loss)
        string += "----------------------------------------------------------------\n"
        string += f"           operations saved: {round((self.ops_saved/self.total_ops)*100, 3)}%\n"
        string += f"           with accuracy of: {self.final_acc}%\n"
        string += "================================================================\n"
        return string
        
class BaselineResultRc():
    def __init__(self, init_acc, baseline_acc, ops_saved, tot_ops, ps, net_name, dataset_name):
        self.filename = f'BS_{net_name}_ps{ps}_os{round((ops_saved/tot_ops)*100, 3)}_acc{baseline_acc}'
        self.baseline_acc = baseline_acc
        self.ops_saved = ops_saved
        self.total_ops = tot_ops
        self.patch_size = ps
        self.network = net_name
        self.dataset_name = dataset_name
        self.init_acc = init_acc
        
    def __str__(self):
        string =  "================================================================\n"
        string += " BASELINE FOR:  {:>15} {:>10}\n".format(self.network, self.dataset_name)
        string += "{:>20}: {}\n".format("PATCH SIZE", self.patch_size)
        string += "{:>20}: {}\n".format("TEST SET SIZE", cfg.TEST_SET_SIZE)
        string += "{:>20}: {}\n".format("INITIAL ACC", self.init_acc)
        string += "----------------------------------------------------------------\n"
        string += f"           operations saved: {round((self.ops_saved/self.total_ops)*100, 3)}% \n"
        string += f"           with accuracy of: {self.baseline_acc}% \n"
        string += "================================================================\n"
        return string
        

