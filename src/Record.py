# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 11:29:28 2018

@author: Inna
"""
from bitarray import bitarray
import numpy as np
import pickle
import math
import time
import os
import csv

Resnet18_layers_layout = [(64, 32, 32),(64, 32, 32), (128, 16, 16), (256, 8, 8), (512, 4, 4)]
max_granularity = 0
uniform_filters = 1
uniform_patch = 2

def bitmasks(n,m):
    if m < n:
        if m > 0:
            for x in bitmasks(n-1,m-1):
                yield bitarray([1]) + x
            for x in bitmasks(n-1,m):
                yield bitarray([0]) + x
        else:
            yield n * bitarray('0')
    else:
        yield n * bitarray('1')
        
def patches(patch_size, min_ones, max_ones):
    n = patch_size*patch_size
    for m in range(min_ones, max_ones):
        for patch in bitmasks(n,m):
            yield np.array(patch, dtype=int).reshape((patch_size,patch_size))
            
def all_patches_array(patch_size, min_ones, max_ones):
    all_patches = np.zeros((patch_size,patch_size,1))
    for p in patches(patch_size, min_ones, max_ones):
        all_patches = np.append(all_patches,p[:, :, np.newaxis], axis=2)
    return all_patches[:,:,1:]

def actual_patch_size(N, M, patch_size, max_gra):
    granularity = (N*M)/(patch_size*patch_size)
    new_patch_size = patch_size
    while granularity > max_gra:
        new_patch_size += patch_size
        granularity = (N*M)/(new_patch_size*new_patch_size)
    return new_patch_size

class Record():
    '''
    results[layer][channel][patch][pattern] = (op,acc)
    '''
    def __init__(self,layers_layout, max_gra, gen_patches, mode,initial_acc, *argv):
        self.mode = mode
        self.max_gra = max_gra
        self.no_of_layers = len(layers_layout)
        self.no_of_channels = [l[0] for l in layers_layout]
        if gen_patches:
            self.all_patterns = all_patches_array(argv[0],argv[1],argv[2]) # [patch_size, min_ones, max_ones]
        else:
            self.all_patterns = argv[0]
        
        patch_size = self.all_patterns.shape[0]
        self.patch_sizes = [actual_patch_size(l[1], l[2], patch_size, max_gra) for l in layers_layout]
        self.no_of_patches = [math.ceil(layers_layout[idx][1]/self.patch_sizes[idx])* \
                         math.ceil(layers_layout[idx][2]/self.patch_sizes[idx]) for idx in range(self.no_of_layers)]
            
        if mode == uniform_filters:
           self.no_of_channels = [1]*self.no_of_layers
           self.filename = 'uniform_filters_'
        elif mode == uniform_patch:
           self.no_of_patches = [1]*self.no_of_layers
           self.filename = 'uniform_patch_'
        else:
           self.filename = 'max_granularity_'
           
        self.filename += 'acc'+ str(initial_acc) +'_mg'+str(self.max_gra)+'_'+str(int(time.time())) 
        
        self.results = []
        for l in range(self.no_of_layers):
            layer = []
            for k in range(self.no_of_channels[l]):
                channel = []
                for j in range(self.no_of_patches[l]):
                        patch = []
                        for i in range(self.all_patterns.shape[2]):
                                patch.append(None)
                        channel.append(patch)
                layer.append(channel)
            self.results.append(layer)

            
    def addRecord(self, op, tot_op, acc, layer, channel=0, patch_idx=0, pattern_idx=0):
        if self.mode == uniform_filters:
           channel = 0
        elif self.mode == uniform_patch:
           patch_idx = 0
         
        self.results[layer][channel][patch_idx][pattern_idx] = (op,tot_op,acc)
    
    def find_resume_point(self):
        for layer in range(self.no_of_layers):
            for channel in range(self.no_of_channels[layer]):
                for patch_idx in range(self.no_of_patches[layer]):
                    for pattern_idx in range(self.all_patterns.shape[2]):
                        if self.results[layer][channel][patch_idx][pattern_idx] is None:
                            return layer, channel, patch_idx, pattern_idx
        
    def save_to_csv(self, path = './results'):
        out_path = os.path.join(path, self.filename + ".csv")
        with open(out_path, 'w', newline='') as f:
            csv.writer(f).writerow(['layer', 'channel', 'patch number', 'pattern number', 'operations saved', 'total operations','accuracy'])
            for l in range(self.no_of_layers):
                for k in range(self.no_of_channels[l]):
                    for j in range(self.no_of_patches[l]):
                        for i in range(self.all_patterns.shape[2]):
                            if self.results[l][k][j][i] is not None:
                                csv.writer(f).writerow([l,k,j,i,self.results[l][k][j][i][0], \
                                                                  self.results[l][k][j][i][1],\
                                                                  self.results[l][k][j][i][2]])
        
    
            
        
def save_to_file(record, use_default=True,path='./data/results', filename=''):
    if use_default:
        filename = record.filename
    if not os.path.isdir(path):
        os.mkdir(path)
    outfile = open(os.path.join(path,filename),'wb')
    pickle.dump(record,outfile)
    outfile.close()
        

        
def load_from_file(filename, path='./data/results'):
    infile = open(os.path.join(path,filename),'rb')
    record = pickle.load(infile)
    infile.close()
    return record
        
        
#-----------------Test---------------------
#max_gra = 32*32        
#r = Record(Resnet18_layers_layout,max_gra,True, max_granularity,2,2,3)
#
#ap = all_patches_array(2,2,3)
#r1 = Record(Resnet18_layers_layout,max_gra,False, max_granularity,ap)      
#
#assert(r1.results ==r.results)  
#
#r2 = Record(Resnet18_layers_layout,max_gra,True, uniform_filters,2,2,3) 
#r2.addRecord( 4, 0.98, 0, 0, 0, 0)  
#r2.save_to_csv()    
#           
#save_to_file(r2)
#r3 = load_from_file('uniform_filters_1542549224_mg1024') 
#assert(r2.results ==r3.results)         
    