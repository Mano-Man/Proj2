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
import glob
import re

import Record as rc
import maskfactory as mf
import NeuralNet as net
import Config as cfg
from LayerQuantizier import LayerQuantizier




# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def main():
    nn = net.NeuralNet(cfg.SP_MOCK)
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    _, test_acc, _ = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc}\n')

    if cfg.RESUME_MASK_GEN:
        rcs = rc.load_from_file(cfg.RECORDS_FILENAME, path=cfg.RESULTS_DIR)
        st_point = rcs.find_resume_point()
    else:
        rcs = rc.Record(cfg.LAYER_LAYOUT, cfg.GRAN_THRESH, cfg.GEN_PATTERNS, cfg.MODE, test_acc, cfg.PS, cfg.ONES_RANGE)
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


def training_main():
    nn = net.NeuralNet(cfg.CHOSEN_SP)
    nn.train(cfg.N_EPOCHS)
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')

def find_rec_file(mode): 
    rec_filename = glob.glob(f'{cfg.RESULTS_DIR}ps*{rc.gran_dict[mode]}_*pkl')
    if not rec_filename:
        return None
    else:
        rec_filename.sort(key=os.path.getmtime)
        # print(checkpoints)
        return rec_filename[-1]

def lQ_main():
    in_rec_fn = find_rec_file(rc.uniform_layer) 
    print('==> loading record file from ' + in_rec_fn)
    in_rec = rc.load_from_file(in_rec_fn,path='')
    init_acc = float(re.findall(r'\d+\.\d+', in_rec_fn)[0])
    lQ = LayerQuantizier(in_rec,init_acc-cfg.MAX_ACC_LOSS ,cfg.PS)
    print('==> starting simulation. file will be saved to ' + lQ.output_rec.filename)
    lQ.simulate()
    print('==> finised simulation. file saved to ' + lQ.output_rec.filename)
    
    




if __name__ == '__main__':
    lQ_main()
