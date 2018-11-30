# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
# Torch Libraries:
import torch
import torch.nn



# Utils:
import os
from tqdm import tqdm
from util.data_import import CIFAR10_Test

import Record as rc
import maskfactory as mf
import NeuralNet as net
import Config as cfg
import RecordFinder as rf
from LayerQuantizier import LayerQuantizier
from ChannelQuantizier import ChannelQuantizier




# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def gen_first_lvl_results_main(mode):
    
    rec_filename = rf.find_rec_filename(mode, rf.FIRST_LVL_REC)
    if rec_filename is not None:
        rcs = rc.load_from_file(rec_filename, path='')
        st_point = rcs.find_resume_point()
        if None==st_point:
            return rcs
        
    nn = net.NeuralNet(cfg.SP_MOCK)
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    _, test_acc, _ = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc}\n')
    
    if rec_filename is None:
        rcs = rc.Record(cfg.LAYER_LAYOUT, cfg.GRAN_THRESH, True, mode, test_acc, cfg.PS, cfg.ONES_RANGE)
        st_point = [0] * 4
        rcs.filename = 'ps' + str(cfg.PS) + '_ones' + str(cfg.ONES_RANGE) + '_' + rcs.filename
        
    print('==> Result will be saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
    save_counter = 0
    for layer, channel, patch, pattern_idx, mask in tqdm(mf.gen_masks_with_resume \
                                                                     (cfg.PS, rcs.all_patterns, rcs.mode, rcs.gran_thresh,
                                                                      cfg.LAYER_LAYOUT, resume_params=st_point)):
        sp_list = cfg.SP_MOCK
        sp_list[layer] = (1, cfg.PS, torch.from_numpy(mask))
        nn.update_spatial(sp_list)
        _, test_acc, _ = nn.test(test_gen)
        ops_saved, ops_total = nn.net.num_ops()
        rcs.addRecord(ops_saved.item(), ops_total, test_acc, layer, channel, patch, pattern_idx)

        save_counter += 1
        if save_counter > cfg.SAVE_INTERVAL:
            rc.save_to_file(rcs, True, cfg.RESULTS_DIR)
            save_counter = 0

    rc.save_to_file(rcs, True, cfg.RESULTS_DIR)
    rcs.save_to_csv(cfg.RESULTS_DIR)
    print('==> Result saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
    return rcs


def training_main():
    nn = net.NeuralNet(cfg.CHOSEN_SP)
    nn.train(cfg.N_EPOCHS)
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')
    
def lQ_main(in_rec):
    init_acc = rf.get_init_acc(in_rec.filename)
    lQ_rec_fn =  rf.find_rec_filename(in_rec.mode, rf.lQ_REC)
    if lQ_rec_fn is None:
        lQ = LayerQuantizier(in_rec,init_acc-cfg.MAX_ACC_LOSS ,cfg.PS)
    else:
        lQ = LayerQuantizier(in_rec,init_acc-cfg.MAX_ACC_LOSS ,cfg.PS,None, \
                             rc.load_from_file(lQ_rec_fn, ''))
    lQ.simulate()
    return lQ.output_rec

def by_uniform_layers():
    in_rec = gen_first_lvl_results_main(rc.uniform_layer)
    lQ_rec = lQ_main(in_rec)
    min_acc = rf.get_min_acc(lQ_rec.filename)
    rf.print_best_results(lQ_rec, min_acc)
   
def cQ_main(in_rec):
    init_acc = rf.get_init_acc(in_rec.filename)
    cQ_rec_fn =  rf.find_rec_filename(in_rec.mode,rf.cQ_REC)
    if cQ_rec_fn is None:
        cQ = ChannelQuantizier(in_rec,init_acc-cfg.MAX_ACC_LOSS ,cfg.PS)
    else:
        cQ = ChannelQuantizier(in_rec,init_acc-cfg.MAX_ACC_LOSS ,cfg.PS,None, \
                             rc.load_from_file(cQ_rec_fn, ''))
    cQ.simulate()
    return cQ.output_rec

def by_uniform_patches():
    in_rec = gen_first_lvl_results_main(rc.uniform_patch)
    cQ_rec = cQ_main(in_rec)
    lQ_rec = lQ_main(cQ_rec)
    min_acc = rf.get_min_acc(lQ_rec.filename)
    rf.print_best_results(lQ_rec, min_acc)


if __name__ == '__main__':
    by_uniform_patches() 
