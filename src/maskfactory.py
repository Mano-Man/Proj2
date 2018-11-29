# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:01:16 2018
N,M,C <- dim
@author: Inna
"""



import numpy as np
import math
import Record as rc
 

        
def int2base(x, base):
    digits = []
    while x:
        digits.append(int(x % base))
        x = int(x / base)
    return np.array(digits, dtype=np.intc)
 
def tile_opt(dims, pattern, is3D=True):
    if is3D and len(pattern.shape) == 2:
        pattern = pattern[np.newaxis, :, :]
    rdims = (math.ceil(dims[i]/pattern.shape[i]) for i in range(len(dims)))
    return np.tile(pattern, rdims)
        
            
def uniform_mask2d(N, M, patch_size, patterns, p_start=0):
    N = math.ceil(N/patch_size)
    M = math.ceil(M/patch_size)
    for p_idx in range(p_start, patterns.shape[2]):
        yield p_idx, np.tile(patterns[:,:,p_idx], (N,M))
        
def uniform_all(C,N, M, patch_size, patterns,  p_start=0):
    for p_idx, filt in uniform_mask2d(N, M, patch_size, patterns, p_start):
        yield p_idx, np.repeat(filt[np.newaxis, :, :],C,axis=0)
        
def uniform_layer(C, mask2d):
    return np.repeat(mask2d[np.newaxis, :, :],C,axis=0)
        
def random_varied_mask(N, M, patch_size, min_ones, max_ones):
    patch_n = math.ceil(N/patch_size)
    patch_m = math.ceil(M/patch_size)
    mask = np.zeros((patch_n*patch_size,patch_m*patch_size))
    #no_of_patches = patch_n*patch_m
    all_patches = rc.all_patches_array(patch_size, min_ones, max_ones);
    patch_options = all_patches.shape[2]
    # so we have patch_options^no_of_pathces
    # for opt = range(patch_options^no_of_pathces)
    #   convers opt to number in base patch_options and do:
    patch_tiling  = np.random.randint(0,patch_options,(patch_n, patch_m))
    for ii in range(patch_n):
        for jj in range(patch_m):
            mask[ii*patch_size:ii*patch_size+patch_size, jj*patch_size:jj*patch_size+patch_size] = all_patches[:,:,patch_tiling[ii,jj]]
    return mask

def gen_varied_masks(N, M, patch_size, min_ones, max_ones,max_opt):
    patch_n = math.ceil(N/patch_size)
    patch_m = math.ceil(M/patch_size)
    mask = np.zeros((patch_n*patch_size,patch_m*patch_size))
    no_of_patches = patch_n*patch_m
    all_patches = rc.all_patches_array(patch_size, min_ones, max_ones);
    patch_options = all_patches.shape[2]
    # so we have patch_options^no_of_pathces <- it's too much!
    #for opt in np.random.randint(0, patch_options**no_of_patches, max_opt):
    all_options = patch_options**no_of_patches
    if max_opt==0:
        max_opt = all_options
    for opt in range(max_opt):
        if opt >= max_opt:
            return
    #   convers opt to number in base patch_options and do:
        patch_tiling  = int2base(opt, patch_options)
        patch_tiling = np.resize(patch_tiling,no_of_patches)
        patch_tiling = patch_tiling.reshape((patch_n, patch_m))
        for ii in range(patch_n):
            for jj in range(patch_m):
                mask[ii*patch_size:ii*patch_size+patch_size, jj*patch_size:jj*patch_size+patch_size] = all_patches[:,:,patch_tiling[ii,jj]]
        yield mask
        
def change_one_patch2d(mask, patch_n, patch_m, patch_size, p):
    mask[patch_n*patch_size:patch_n*patch_size+patch_size, patch_m*patch_size:patch_m*patch_size+patch_size] = p
    return mask
    
def change_one_patch3d(mask, patch_n, patch_m, patch_size, p, c):
    mask[c, patch_n*patch_size:patch_n*patch_size+patch_size, patch_m*patch_size:patch_m*patch_size+patch_size] = p
    return mask
    
        
def one_patch_diff2d(N, M, patch_size, patterns):
    patch_n = math.ceil(N/patch_size)
    patch_m = math.ceil(M/patch_size)
    for p_idx in patterns.shape[2]:
        for ii in range(patch_n):
            for jj in range(patch_m):
                mask = np.ones((patch_n*patch_size,patch_m*patch_size))
                yield ii*patch_n+jj, p_idx, change_one_patch2d(mask, ii, jj, patch_size, patterns[:,:,p_idx])
                
    
def one_patch_diff3d(C, N, M, patch_size, patterns, gran_thresh):
    new_patch_size = rc.actual_patch_size(N, M, patch_size, gran_thresh)
    patch_n = math.ceil(N/new_patch_size)
    patch_m = math.ceil(M/new_patch_size)
    for p_idx, p in uniform_mask2d(new_patch_size, new_patch_size, patch_size, patterns): 
        for cc in range(C):
            for ii in range(patch_n):
                for jj in range(patch_m):
                    mask = np.ones((C, patch_n*new_patch_size,patch_m*new_patch_size))
                    yield cc, ii*patch_n+jj, p_idx, change_one_patch3d(mask, ii, jj, new_patch_size, p, cc)
                
def one_patch_diff3d_uniform_filters(C, N, M, patch_size, patterns):
    for patch, p_idx, mask2d in one_patch_diff2d(N, M, patch_size, patterns):
        yield patch, p_idx, uniform_layer(C, mask2d)
                    
def one_patch_diff3d_uniform_patch(C, N, M, patch_size, patterns):
    patch_n = math.ceil(N/patch_size)
    patch_m = math.ceil(M/patch_size)
    for p_idx ,mask2d in uniform_mask2d(N, M, patch_size, patterns):
        for cc in range(C):
            mask = np.ones((C, patch_n*patch_size,patch_m*patch_size))
            mask[cc,:,:] = mask2d
            yield cc, p_idx, mask
            
            
def gen_masks(patch_size,patterns, mode, gran_thresh, layer_layout):
    layer = -1;   
    for C, N, M in  layer_layout:
        layer += 1
        if mode == rc.uniform_filters:
            for patch, p_idx, mask in one_patch_diff3d_uniform_filters(C, N, M, patch_size, patterns):
                yield layer, 0 , patch, p_idx, mask 
        elif mode == rc.uniform_patch:
            for channel, p_idx, mask in one_patch_diff3d_uniform_patch(C, N, M, patch_size, patterns):
                yield layer, channel, 0, p_idx, mask
        else: #mode=='max_granularity'
            for channel, patch, p_idx, mask in one_patch_diff3d(C, N, M, patch_size, patterns, gran_thresh):
                yield layer, channel, patch, p_idx, mask
                
def gen_masks_with_resume(patch_size,patterns, mode, gran_thresh, layer_layout, resume_params=[0,0,0,0], mask_type=np.float32):
    for layer in  range(resume_params[0], len(layer_layout)):
        C, N, M = layer_layout[layer]
        patch_n = math.ceil(N/patch_size)
        patch_m = math.ceil(M/patch_size)
        if mode == rc.uniform_filters:
            ii_start = int(resume_params[2]/patch_n)
            jj_start = resume_params[2] - ii_start*patch_n
            for ii in range(ii_start, patch_n):
                resume_params[2] = 0 
                ii_start = 0
                for jj in range(jj_start, patch_m):
                    jj_start = 0
                    for p_idx in range(resume_params[3], patterns.shape[2]):
                        resume_params[3] = 0
                        mask = np.ones((patch_n*patch_size,patch_m*patch_size), dtype=mask_type)
                        yield layer, 0, ii*patch_n+jj, p_idx, uniform_layer(C, change_one_patch2d(mask, ii, jj, patch_size, patterns[:,:,p_idx]))
        elif mode == rc.uniform_patch:
            for channel in range(resume_params[1], C):
                resume_params[1] = 0
                for p_idx ,mask2d in uniform_mask2d(N, M, patch_size, patterns, p_start=resume_params[3]):
                    resume_params[3] = 0
                    mask = np.ones((C, patch_n*patch_size,patch_m*patch_size) , dtype=mask_type)
                    mask[channel,:,:] = mask2d
                    yield layer, channel, 0, p_idx, mask
        elif mode == rc.uniform_layer:
            for p_idx, mask in uniform_all(C,N, M, patch_size, patterns,  p_start=resume_params[3]):
                resume_params[3] = 0
                yield layer, 0, 0, p_idx, mask
        else: #mode=='max_granularity'
            new_patch_size = rc.actual_patch_size(N, M, patch_size, gran_thresh)
            patch_n = math.ceil(N/new_patch_size)
            patch_m = math.ceil(M/new_patch_size)
            ii_start = int(resume_params[2]/patch_n)
            jj_start = resume_params[2] - ii_start*patch_n
            for channel in range(resume_params[1], C):
                resume_params[1] = 0
                for ii in range(ii_start, patch_n):
                    resume_params[2] = 0 
                    ii_start = 0
                    for jj in range(jj_start, patch_m):
                        jj_start = 0
                        for p_idx, p in uniform_mask2d(new_patch_size, new_patch_size, patch_size, patterns, p_start=resume_params[3]):
                            resume_params[3] = 0
                            mask = np.ones((C, patch_n*new_patch_size,patch_m*new_patch_size) , dtype=mask_type)
                            yield layer, channel, ii*patch_n+jj, p_idx, change_one_patch3d(mask, ii, jj, new_patch_size, p, channel)
            
        
    
#--------------------Test------------------------------------------------------
                            
#layer_layout = rc.Resnet18_layers_layout
#patch_size = 2
#sp_list = []
#device0 = 'cuda' if torch.cuda.is_available() else 'cpu'
#for C, N, M in  layer_layout:
#    sp_list.append((patch_size, torch.ones([C,N,M],device=device0)))
#records = rc.Record(layer_layout,32*32,True, rc.uniform_layer,93.44,2,2,3)
#gen = gen_masks_with_resume(patch_size, records.all_patterns, records.mode, \
#                    records.gran_thresh,layer_layout)
#layer, channel, patch, pattern_idx, mask = next(gen)