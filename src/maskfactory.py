# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import math
from Record import Mode, actual_patch_size

# ----------------------------------------------------------------------------------------------------------------------
#                                          Util Functions for Generating Masks
# ----------------------------------------------------------------------------------------------------------------------
 
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
        yield p_idx, uniform_layer(C,filt)
        
def uniform_layer(C, mask2d):
    return np.repeat(mask2d[np.newaxis, :, :],C,axis=0)
       
def get_patch_indexes(index, N, patch_size):
        patch_n = math.ceil(N/patch_size)
        ii_start = int(index/patch_n)
        jj_start = index - ii_start*patch_n
        return ii_start, jj_start

def change_one_patch2d(mask, patch_n, patch_m, patch_size, p):
    mask[patch_n*patch_size:patch_n*patch_size+patch_size, patch_m*patch_size:patch_m*patch_size+patch_size] = p
    return mask
    
def change_one_patch3d(mask, patch_n, patch_m, patch_size, p, c):
    mask[c, patch_n*patch_size:patch_n*patch_size+patch_size, patch_m*patch_size:patch_m*patch_size+patch_size] = p
    return mask
    
def gen_masks_with_resume(patch_size,patterns, mode, gran_thresh, layer_layout, resume_params=[0,0,0,0], mask_type=np.float32):
    for layer in  range(resume_params[0], len(layer_layout)):
        C, N, M = layer_layout[layer]
        patch_n = math.ceil(N/patch_size)
        patch_m = math.ceil(M/patch_size)
        if mode == Mode.UNIFORM_FILTERS:
            ii_start, jj_start = get_patch_indexes(resume_params[2],N,patch_size)
            for ii in range(ii_start, patch_n):
                resume_params[2] = 0 
                ii_start = 0
                for jj in range(jj_start, patch_m):
                    jj_start = 0
                    for p_idx in range(resume_params[3], patterns.shape[2]):
                        resume_params[3] = 0
                        mask = np.ones((patch_n*patch_size,patch_m*patch_size), dtype=mask_type)
                        yield layer, 0, ii*patch_n+jj, p_idx, uniform_layer(C, change_one_patch2d(mask, ii, jj, patch_size, patterns[:,:,p_idx]))
        elif mode == Mode.UNIFORM_PATCH:
            for channel in range(resume_params[1], C):
                resume_params[1] = 0
                for p_idx ,mask2d in uniform_mask2d(N, M, patch_size, patterns, p_start=resume_params[3]):
                    resume_params[3] = 0
                    mask = np.ones((C, patch_n*patch_size,patch_m*patch_size) , dtype=mask_type)
                    mask[channel,:,:] = mask2d
                    yield layer, channel, 0, p_idx, mask
        elif mode == Mode.UNIFORM_LAYER:
            for p_idx, mask in uniform_all(C,N, M, patch_size, patterns,  p_start=resume_params[3]):
                resume_params[3] = 0
                yield layer, 0, 0, p_idx, mask
        else: #mode=='max_granularity'
            new_patch_size = actual_patch_size(N, M, patch_size, gran_thresh)
            patch_n = math.ceil(N/new_patch_size)
            patch_m = math.ceil(M/new_patch_size)
            ii_start, jj_start = get_patch_indexes(resume_params[2],N,new_patch_size)
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

