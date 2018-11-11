# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 15:01:16 2018
N,M,C <- dim
@author: Inna
"""

from bitarray import bitarray
import numpy as np
import math

 
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
        
def int2base(x, base):
    digits = []
    while x:
        digits.append(int(x % base))
        x = int(x / base)
    return np.array(digits, dtype=np.intc)
 
    
def patches(patch_size, min_ones, max_ones):
    n = patch_size*patch_size;
    for m in range(min_ones, max_ones):
        for patch in bitmasks(n,m):
            yield np.array(patch, dtype=int).reshape((patch_size,patch_size))
            
def all_patches_array(patch_size, min_ones, max_ones):
    all_patches = np.zeros((patch_size,patch_size,1))
    for p in patches(patch_size, min_ones, max_ones):
        all_patches = np.append(all_patches,p[:, :, np.newaxis], axis=2)
    return all_patches[:,:,1:]
            
def uniform_mask2d(N, M, patch_size, min_ones, max_ones):
    N = math.ceil(N/patch_size)
    M = math.ceil(M/patch_size)
    for p in patches(patch_size, min_ones, max_ones):
        yield np.tile(p, (N,M))
        
def uniform_all(C,N, M, patch_size, min_ones, max_ones):
    for filt in uniform_mask2d(N, M, patch_size, min_ones, max_ones):
        yield np.repeat(filt[:, :, np.newaxis],C,axis=2)
        
def uniform_layer(C, mask2d):
    return np.repeat(mask2d[:, :, np.newaxis],C,axis=2)
        
def random_varied_mask(N, M, patch_size, min_ones, max_ones):
    patch_n = math.ceil(N/patch_size)
    patch_m = math.ceil(M/patch_size)
    mask = np.zeros((patch_n*patch_size,patch_m*patch_size))
    #no_of_patches = patch_n*patch_m
    all_patches = all_patches_array(patch_size, min_ones, max_ones);
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
    all_patches = all_patches_array(patch_size, min_ones, max_ones);
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
    mask[patch_n*patch_size:patch_n*patch_size+patch_size, patch_m*patch_size:patch_m*patch_size+patch_size, c] = p
    return mask
    
        
def one_patch_diff2d(N, M, patch_size, min_ones, max_ones):
    patch_n = math.ceil(N/patch_size)
    patch_m = math.ceil(M/patch_size)
    for p in patches(patch_size, min_ones, max_ones):
        for ii in range(patch_n):
            for jj in range(patch_m):
                mask = np.ones((patch_n*patch_size,patch_m*patch_size))
                yield change_one_patch2d(mask, ii, jj, patch_size, p)
                
    
def one_patch_diff3d(C, N, M, patch_size, min_ones, max_ones):
    patch_n = math.ceil(N/patch_size)
    patch_m = math.ceil(M/patch_size)
    for p in patches(patch_size, min_ones, max_ones):
        for cc in range(C):
            for ii in range(patch_n):
                for jj in range(patch_m):
                    mask = np.ones((patch_n*patch_size,patch_m*patch_size, C))
                    yield change_one_patch3d(mask, ii, jj, patch_size, p, cc)
                
def one_patch_diff3d_uniform_filters(C, N, M, patch_size, min_ones, max_ones):
    for mask2d in one_patch_diff2d(N, M, patch_size, min_ones, max_ones):
        yield uniform_layer(C, mask2d)
                    
def one_patch_diff3d_uniform_patch(C, N, M, patch_size, min_ones, max_ones):
    patch_n = math.ceil(N/patch_size)
    patch_m = math.ceil(M/patch_size)
    for mask2d in uniform_mask2d(N, M, patch_size, min_ones, max_ones):
        for cc in range(C):
            mask = np.ones((patch_n*patch_size,patch_m*patch_size, C))
            mask[:,:,cc] = mask2d
            yield mask
                
    
        
        
    

count =0
mask = 0
for m in one_patch_diff3d_uniform_patch(64, 32, 32, 2, 2, 3):
    mask = m
    count = count + 1