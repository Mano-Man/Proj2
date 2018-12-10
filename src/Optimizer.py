# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn

import os
from tqdm import tqdm
import random
import time

#from util.data_import import CIFAR10_Test, CIFAR10_shape
from RecordFinder import RecordFinder, RecordType
from NeuralNet import NeuralNet
from Record import Mode, Record, BaselineResultRc, load_from_file, save_to_file
from PatchQuantizier import PatchQuantizier
from ChannelQuantizier import ChannelQuantizier
from LayerQuantizier import LayerQuantizier
import maskfactory as mf
import Config as cfg


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class Optimizer:
    def __init__(self, patch_size, ones_range, gran_thresh, max_acc_loss, calc_init_acc=True, init_acc=None):
        self.record_finder = RecordFinder(cfg.NET.__name__, patch_size, ones_range, gran_thresh, max_acc_loss)
        self.nn = NeuralNet()
        self.test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, max_dataset_size=cfg.TEST_SET_SIZE, download=cfg.DO_DOWNLOAD)
        self.test_set_size = cfg.TEST_SET_SIZE
        if calc_init_acc:
            _, test_acc, correct = self.nn.test(self.test_gen)
            print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
            self.init_acc = test_acc  # TODO - Fix initialize bug 
        else:
            self.init_acc = init_acc
        self.ps = patch_size
        self.max_acc_loss = max_acc_loss
        self.gran_thresh = gran_thresh
        self.ones_range = ones_range
        self.full_net_run_time = None

    def base_line_result(self):
        layers_layout = self.nn.net.generate_spatial_sizes(cfg.DATA_SHAPE())
        self._init_nn()

        sp_list = [None] * len(layers_layout)
        for layer, layer_mask in enumerate(mf.base_line_mask(layers_layout, self.ps)):
            sp_list[layer] = torch.from_numpy(layer_mask)
        self.nn.net.strict_mask_update(update_ids=list(range(len(layers_layout))), masks=sp_list)

        _, test_acc, _ = self.nn.test(self.test_gen)
        ops_saved, ops_total = self.nn.net.num_ops()
        bl_rec = BaselineResultRc(self.init_acc, test_acc, ops_saved, ops_total, self.ps, cfg.NET.__name__, cfg.DATA_NAME)
        print(bl_rec)
        save_to_file(bl_rec, True, cfg.RESULTS_DIR)

    def _quantizier_main(self, rec_type, in_rec):
        if rec_type == RecordType.lQ_RESUME:
            resume_param_path = self.record_finder.find_rec_filename(in_rec.mode, RecordType.lQ_RESUME)
            quantizier = LayerQuantizier(in_rec, self.init_acc, self.max_acc_loss, self.ps, self.ones_range,
                                         resume_param_path)
        else:
            q_rec_fn = self.record_finder.find_rec_filename(in_rec.mode, rec_type)
            Quantizier = PatchQuantizier if rec_type == RecordType.pQ_REC else ChannelQuantizier
            if q_rec_fn is None:
                quantizier = Quantizier(in_rec, self.init_acc, self.max_acc_loss, self.ps)
            else:
                quantizier = Quantizier(in_rec, self.init_acc, self.max_acc_loss, self.ps, 
                                        load_from_file(q_rec_fn, ''))
        if not quantizier.is_finised():
            self.nn.net.reset_spatial()
            quantizier.simulate(self.nn, self.test_gen)
        if RecordType.lQ_RESUME == rec_type:
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
    
    def print_runtime_eval(self):        
        print(f"================================================================")
        print(f"----------------------------------------------------------------")
        print(f"                      NET: {cfg.NET.__name__}")
        print(f"                  DATASET: {cfg.DATA_NAME}")
        print(f"               PATCH SIZE: {self.ps}")
        print(f"                     ONES: {self.ones_range[0]}-{self.ones_range[1]-1}")
        print(f"              GRANULARITY: {self.gran_thresh}")
        print(f"            TEST SET SIZE: {self.test_set_size}")
        print(f"----------------------------------------------------------------")
        for mode in Mode:
            no_of_runs, run_times = self.eval_run_time(mode)
            total_run_time = (no_of_runs[0] * run_times[0] + no_of_runs[1] * run_times[0] + no_of_runs[2] * run_times[1]) / (60 * 60)
            if total_run_time > 24:
                total_run_time = round(total_run_time / 24, 2)
                total_run_time_units = 'days'
            else:
                total_run_time = round(total_run_time, 2)
                total_run_time_units = 'hours'
            print("    {}    {:>25} [{}]".format(mode, total_run_time, total_run_time_units))
            print(f"----------------------------------------------------------------")
            print(f"         iters 1st lvl         iters 2nd lvl          iters lQ ")
            print("number {:>15} {:>21} {:>17}".format(no_of_runs[0], no_of_runs[1], no_of_runs[2]))
            print("time   {:>15} {:>21} {:>17}".format(round(no_of_runs[0] * run_times[0]),
                                                       round(no_of_runs[1] * run_times[0]),
                                                       round(no_of_runs[2] * run_times[1])))
            print(f"\nsec per iter    ")
            print(f"        1st/2nd lvl: {run_times[0]}")
            print(f"        lQ: {run_times[1]}")
    
            print(f"----------------------------------------------------------------")
        print(f"================================================================")

    def eval_run_time(self, mode, no_of_tries=5):
        layers_layout = self.nn.net.generate_spatial_sizes(cfg.DATA_SHAPE())
        recs_first_lvl = Record(layers_layout, self.gran_thresh, True, mode, self.init_acc, self.ps, self.ones_range)
        first_lvl_runs = recs_first_lvl.size

        self.nn.net.reset_spatial()
        run_time_for_iter = 0
        for idx in range(no_of_tries):
            layer = random.randint(0, recs_first_lvl.no_of_layers - 1)
            channel = random.randint(0, recs_first_lvl.no_of_channels[layer] - 1)
            patch = random.randint(0, recs_first_lvl.no_of_patches[layer] - 1)
            pattern_idx = random.randint(0, recs_first_lvl.no_of_patterns[layer] - 1)
            pattern = recs_first_lvl.all_patterns[:, :, pattern_idx]
            mask = mf.get_specific_mask(layers_layout[layer], channel, patch, pattern,
                                        recs_first_lvl.patch_sizes[layer], mode)
            st_time = time.time()
            self.nn.net.reset_spatial()
            self.nn.net.strict_mask_update(update_ids=[layer], masks=[torch.from_numpy(mask)])
            _, test_acc, _ = self.nn.test(self.test_gen)
            end_time = time.time()
            run_time_for_iter += (end_time - st_time)

        run_time_for_iter = run_time_for_iter / no_of_tries

        if mode == Mode.MAX_GRANULARITY:
            second_lvl_runs = sum([recs_first_lvl.no_of_channels[l] * recs_first_lvl.no_of_patterns[l] for l in
                                   range(recs_first_lvl.no_of_layers)])
            second_lvl_runs += sum(recs_first_lvl.no_of_patterns)
            lQ_runs = sum(recs_first_lvl.no_of_patterns)
        elif mode == Mode.UNIFORM_LAYER:
            second_lvl_runs = 0
            lQ_runs = sum(recs_first_lvl.no_of_patterns)
        else:  # Mode.UNIFORM_PATCH or mode==Mode.UNIFORM_FILTERS
            second_lvl_runs = sum(recs_first_lvl.no_of_patterns)
            lQ_runs = second_lvl_runs  # sum(recs_first_lvl.no_of_patterns)

        no_of_runs = (first_lvl_runs, second_lvl_runs, lQ_runs)
        run_times = (round(run_time_for_iter, 3), self.get_full_net_run_time(no_of_tries))
        return no_of_runs, run_times

    def get_full_net_run_time(self, no_of_tries):
        if self.full_net_run_time is None:
            self.nn.net.reset_spatial()
            self.nn.net.fill_masks_to_val(1)
            self.full_net_run_time = 0
            for idx in range(no_of_tries):
                st_time = time.time()
                _, test_acc, _ = self.nn.test(self.test_gen)
                end_time = time.time()
                self.full_net_run_time += (end_time - st_time)
            self.full_net_run_time = round(self.full_net_run_time / no_of_tries, 3)
        return self.full_net_run_time

    def gen_first_lvl_results(self, mode):
        rec_filename = self.record_finder.find_rec_filename(mode, RecordType.FIRST_LVL_REC)
        if rec_filename is not None:
            rcs = load_from_file(rec_filename, path='')
            st_point = rcs.find_resume_point()
            if st_point is None:
                return rcs

        layers_layout = self.nn.net.generate_spatial_sizes(cfg.DATA_SHAPE())
        self.nn.net.reset_spatial()

        if rec_filename is None:
            rcs = Record(layers_layout, self.gran_thresh, True, mode, self.init_acc, self.ps, self.ones_range)
            st_point = [0] * 4

        print('==> Result will be saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
        save_counter = 0
        for layer, channel, patch, pattern_idx, mask in tqdm(
                mf.gen_masks_with_resume(self.ps, rcs.all_patterns, rcs.mode, rcs.gran_thresh, layers_layout,
                                         resume_params=st_point)):
            self.nn.net.strict_mask_update(update_ids=[layer], masks=[torch.from_numpy(mask)])

            _, test_acc, _ = self.nn.test(self.test_gen)
            ops_saved, ops_total = self.nn.net.num_ops()
            self.nn.net.reset_spatial()
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
        # TODO - Remove this
        self.nn.net.disable_spatial_layers(list(range(len(self.nn.net.generate_spatial_sizes(cfg.DATA_SHAPE())))))
        self.nn.net.initialize_spatial_layers(cfg.DATA_SHAPE(), cfg.BATCH_SIZE, self.ps)
        _, test_acc, correct = self.nn.test(self.test_gen)
        print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
        self.nn.net.reset_spatial()
        assert test_acc == self.init_acc, f'starting accuracy does not match! curr_acc:{test_acc}, prev_acc{self.init_acc}'



