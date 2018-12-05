# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 22:21:34 2018

@author: Inna
"""
import glob
import os
import re
import Record as rc
import Config as cfg

#Record types
FIRST_LVL_REC = 0
pQ_REC = 1
cQ_REC = 2
lQ_REC = 3
FINAL_RESULT_REC = 4
lQ_RESUME = 5

#def print_best_results(lQ_rec, min_acc, num=1):
#    st_point = lQ_rec.find_resume_point
#    if st_point is not None:
#        lQ_rec.fill_empty()
#    res = lQ_rec.gen_pattern_lists(min_acc)[0][0][0]
#    print(f'==> Here are the best {num} results:')
#    if len(res) < num:
#        num = len(res)
#    for idx in range(num):
#        ops_saved, tot_ops, acc = lQ_rec.results[0][0][0][res[idx][0]]
#        if ops_saved < 0 :
#            break
#        print(f'{idx+1}. operations saved: {round((ops_saved/tot_ops)*100, 3)}% with accuracy of: {acc}%')
#
#    f_rec = rc.FinalResultRc(lQ_rec.results[0][0][0][res[0][0]][2], \
#                             lQ_rec.results[0][0][0][res[0][0]][0], \
#                             lQ_rec.results[0][0][0][res[0][0]][1],lQ_rec.mode, \
#                             lQ_rec.all_patterns[res[0][0]], cfg.PS, cfg.MAX_ACC_LOSS,\
#                             cfg.ONES_RANGE, cfg.NET.__name__)
#    rc.save_to_file(f_rec,True,cfg.RESULTS_DIR)
#    print(f'Best Result saved to: ' + f_rec.filename)

def print_result(mode):
    f_rec_fn = find_rec_filename(mode,FINAL_RESULT_REC)
    if f_rec_fn is None:
        return
    f_rec = rc.load_from_file(f_rec_fn, path='')
    print(f'==> Result for {f_rec.network},    mode:{rc.gran_dict[f_rec.mode]}:  \
          operations saved: {round((f_rec.ops_saved/f_rec.total_ops)*100, 3)}% with accuracy of:\
          {f_rec.final_acc}%')
        
def find_rec_file_by_time(regex, check_max_acc_loss=False): 
    rec_filename = glob.glob(f'{cfg.RESULTS_DIR}{regex}')
    if check_max_acc_loss:
        rec_filename[:] = [fn for fn in rec_filename if does_max_acc_loss_match(fn)]
    if not rec_filename:
        return None
    else:
        rec_filename.sort(key=os.path.getmtime)
        return rec_filename[-1]

def first_lvl_regex(mode):
    return f'ps{cfg.PS}_ones{cfg.ONES_RANGE[0]}x{cfg.ONES_RANGE[1]}_{rc.gran_dict[mode]}_acc*_mg{cfg.GRAN_THRESH}_*pkl'

def get_min_acc(fn):
    return float(re.findall(r'\d+\.\d+', fn)[0])

def get_init_acc(fn):
    return float(re.findall(r'\d+\.\d+', fn)[-1])

def does_max_acc_loss_match(fn):
    max_acc_loss_split = str(cfg.MAX_ACC_LOSS).split('.')
    round_len = len(max_acc_loss_split[-1])
    if len(max_acc_loss_split) == 1:
        round_len = 0
    return (round(get_init_acc(fn)-get_min_acc(fn), round_len)==cfg.MAX_ACC_LOSS)

def cQ_regex(mode):
    regex = f'ChannelQ_ma*_'
    #if mode==rc.uniform_patch:
    regex += first_lvl_regex(mode)
    return regex

def lQ_regex(mode):
    regex = f'LayerQ_ma*_'
    if mode==rc.uniform_patch:
        regex += cQ_regex(mode)
    elif mode==rc.uniform_filters:
        regex += pQ_regex(mode)
    else:
        regex += first_lvl_regex(mode)
    return regex

def lQ_resume_regex(mode):
    return 'RP_'+lQ_regex(mode)

def pQ_regex(mode):
    return (f'PatchQ_ma*_' + first_lvl_regex(mode))

def final_rec_regex(mode):
    return f'FR_{cfg.NET.__name__}_ps{cfg.PS}_ones{cfg.ONES_RANGE[0]}x{cfg.ONES_RANGE[1]}_{rc.gran_dict[mode]}_ma{cfg.MAX_ACC_LOSS}'
    
def find_rec_filename(mode, record_type):
    if record_type==FIRST_LVL_REC:
        return find_rec_file_by_time(first_lvl_regex(mode))  
    elif record_type==cQ_REC:
        return find_rec_file_by_time(cQ_regex(mode),True)
    elif record_type==lQ_REC:
        return find_rec_file_by_time(lQ_regex(mode),True)
    elif record_type==lQ_RESUME:
        return find_rec_file_by_time(lQ_resume_regex(mode),True)
    elif record_type==pQ_REC:
        return find_rec_file_by_time(pQ_regex(mode), True)
    elif record_type==FINAL_RESULT_REC:
        return find_rec_file_by_time(final_rec_regex(mode))
    else:
        return None
      