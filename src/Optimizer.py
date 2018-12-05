# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn

import os
from tqdm import tqdm

from util.data_import import CIFAR10_Test, CIFAR10_shape
from RecordFinder import RecordFinder, RecordType
from NeuralNet import NeuralNet
from Record import Mode, Record, load_from_file, save_to_file
from PatchQuantizier import PatchQuantizier
from ChannelQuantizier import ChannelQuantizier
from LayerQuantizier import LayerQuantizier
import maskfactory as mf
import Config as cfg
# ----------------------------------------------------------------------------------------------------------------------
#                                                 Global Config
# ----------------------------------------------------------------------------------------------------------------------
PATCH_SIZE = 2
RANGE_OF_ONES = (1, 3)
GRANULARITY_TH = 32*32
ACC_LOSS = 1.83
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def main():
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    optim = Optimizer(test_gen, PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH, ACC_LOSS)
    optim.by_uniform_layers()
    optim.by_uniform_patches()

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class Optimizer():
    def __init__(self, test_gen, patch_size, ones_range, gran_thresh, max_acc_loss):
        self.record_finder = RecordFinder(cfg.NET.__name__, patch_size, ones_range, gran_thresh, max_acc_loss)
        self.nn = NeuralNet()
        self.test_gen = test_gen
        _, test_acc, correct = self.nn.test(self.test_gen)
        print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
        self.init_acc = test_acc
        self.ps = patch_size
        self.max_acc_loss = max_acc_loss
        self.gran_thresh = gran_thresh
        self.ones_range = ones_range
        
    	
    def _quantizier_main(self,rec_type, in_rec):
        if rec_type==RecordType.lQ_RESUME:
            resume_param_path = self.record_finder.find_rec_filename(in_rec.mode,RecordType.lQ_RESUME)
            quantizier = LayerQuantizier(in_rec, self.init_acc - self.max_acc_loss, self.ps, self.max_acc_loss, self.ones_range, resume_param_path)
        else:
            q_rec_fn = self.record_finder.find_rec_filename(in_rec.mode, rec_type)
            Quantizier = PatchQuantizier if rec_type==RecordType.pQ_REC else ChannelQuantizier
            if q_rec_fn is None:
                quantizier = Quantizier(in_rec, self.init_acc - self.max_acc_loss, self.ps)
            else:
                quantizier = Quantizier(in_rec, self.init_acc - self.max_acc_loss, self.ps, None, 
                                        load_from_file(q_rec_fn, ''))
        if not quantizier.is_finised():
            self._init_nn()
            quantizier.simulate(self.nn, self.test_gen)
        if  RecordType.lQ_RESUME == rec_type:
            return
        return quantizier.output_rec
    
    def by_uniform_layers(self):
        in_rec = self.gen_first_lvl_results(Mode.UNIFORM_LAYER)
        self._quantizier_main(RecordType.lQ_RESUME, in_rec)
        self.record_finder.print_result(Mode.UNIFORM_LAYER)


    def by_uniform_patches(self):
        in_rec = self.gen_first_lvl_results(Mode.UNIFORM_PATCH)
        cQ_rec = self._quantizier_main(RecordType.cQ_REC, in_rec)
        self._quantizier_main(RecordType.lQ_RESUME, cQ_rec)
        self.record_finder.print_result(Mode.UNIFORM_PATCH)


    def by_uniform_filters(self):
        in_rec = self.gen_first_lvl_results(Mode.UNIFORM_FILTERS)
        pQ_rec = self._quantizier_main(RecordType.pQ_REC, in_rec)
        self._quantizier_main(RecordType.lQ_RESUME, pQ_rec)
        self.record_finder.print_result(Mode.UNIFORM_FILTERS)
    
    
    def by_max_granularity(self):
        in_rec = self.gen_first_lvl_results(Mode.MAX_GRANULARITY)
        pQ_rec = self._quantizier_main(RecordType.pQ_REC, in_rec)
        cQ_rec = self._quantizier_main(RecordType.cQ_REC, pQ_rec)
        self._quantizier_main(RecordType.lQ_RESUME, cQ_rec)
        self.record_finder.print_result(Mode.MAX_GRANULARITY)
    
    
    def gen_first_lvl_results(self,mode):
        rec_filename = self.record_finder.find_rec_filename(mode, RecordType.FIRST_LVL_REC)
        if rec_filename is not None:
            rcs = load_from_file(rec_filename, path='')
            st_point = rcs.find_resume_point()
            if st_point == None:
                return rcs
        
        layers_layout = self.nn.net.generate_spatial_sizes(CIFAR10_shape())
        self._init_nn()
    
        if rec_filename is None:  
            rcs = Record(layers_layout, cfg.GRAN_THRESH, True, mode, self.init_acc, self.ps, self.ones_range)
            st_point = [0] * 4
            rcs.filename = f'ps{self.ps}_ones{self.ones_range[0]}x{self.ones_range[1]}_{rcs.filename}'
    
        print('==> Result will be saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
        save_counter = 0
        for layer, channel, patch, pattern_idx, mask in tqdm(mf.gen_masks_with_resume \
                                                                         (self.ps, rcs.all_patterns, rcs.mode,
                                                                          rcs.gran_thresh,
                                                                          layers_layout, resume_params=st_point)):
            self.nn.net.strict_mask_update(update_ids=[layer], masks=[torch.from_numpy(mask)])
    
            _, test_acc, _ = self.nn.test(self.test_gen)
            ops_saved, ops_total = self.nn.net.num_ops()
            rcs.addRecord(ops_saved, ops_total, test_acc, layer, channel, patch, pattern_idx)
    
            save_counter += 1
            if save_counter > cfg.SAVE_INTERVAL:
                save_to_file(rcs, True, cfg.RESULTS_DIR)
                save_counter = 0
    
        save_to_file(rcs, True, cfg.RESULTS_DIR)
        rcs.save_to_csv(cfg.RESULTS_DIR)
        print('==> Result saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
        return rcs
    
    def _init_nn(self):
        self.nn.net.disable_spatial_layers(list(range(len(self.nn.net.generate_spatial_sizes(CIFAR10_shape())))))
        self.nn.net.initialize_spatial_layers(CIFAR10_shape(), cfg.BATCH_SIZE, self.ps)
        _, test_acc, correct = self.nn.test(self.test_gen)
        print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
        assert test_acc == self.init_acc, f'starting accuracy does not match! curr_acc:{test_acc}, prev_acc{test_acc}'
        



if __name__ == '__main__':
    main()


 

