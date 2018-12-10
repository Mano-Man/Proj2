# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn

import os
from tqdm import tqdm
import random
import time

from util.data_import import CIFAR10_Test, CIFAR10_shape
from RecordFinder import RecordFinder, RecordType
from NeuralNet import NeuralNet
from Record import Mode, Record, BaselineResultRc, load_from_file, save_to_file
from PatchQuantizier import PatchQuantizier
from ChannelQuantizier import ChannelQuantizier
from LayerQuantizier import LayerQuantizier
import maskfactory as mf
import Config as cfg

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Optimizer Config
# ----------------------------------------------------------------------------------------------------------------------

PATCH_SIZE = 3
RANGE_OF_ONES = (1, 3)
GRANULARITY_TH = 10
ACC_LOSS = 1.83
TEST_SIZE = cfg.BATCH_SIZE * 8  # This is 1024 - Max for CIFAR10 is 10000  - Better to align it to Batch Size for speed!


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def ac_loss_main():
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, max_dataset_size=cfg.TEST_SET_SIZE, download=cfg.DO_DOWNLOAD)
    for acc_loss in [0, 1, 2, 3, 5]:
        optim = Optimizer(test_gen, CIFAR10_shape(), PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH, acc_loss)
        optim.by_uniform_layers()


def debug_main():
    nn = NeuralNet()
    test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD, max_dataset_size=TEST_SIZE)
    _, test_acc, correct = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
    nn.net.initialize_spatial_layers(cfg.DATA_SHAPE(), cfg.BATCH_SIZE, PATCH_SIZE)

    nn.net.fill_masks_to_val(0)
    test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD, max_dataset_size=TEST_SIZE)
    _, test_acc, correct = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
    nn.net.print_ops_summary()


def main():
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, max_dataset_size=cfg.TEST_SET_SIZE, download=cfg.DO_DOWNLOAD)

    optim = Optimizer(test_gen, CIFAR10_shape(), PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH, ACC_LOSS)
    optim.base_line_result()
    print(f"================================================================")
    print(f"----------------------------------------------------------------")
    print(f"                      NET: {cfg.NET.__name__}")
    print(f"                  DATASET: CIFAR10")
    print(f"               PATCH SIZE: {PATCH_SIZE}")
    print(f"                     ONES: {RANGE_OF_ONES[0]}-{RANGE_OF_ONES[1]-1}")
    print(f"              GRANULARITY: {GRANULARITY_TH}")
    print(f"            TEST SET SIZE: {cfg.TEST_SET_SIZE}")
    print(f"----------------------------------------------------------------")
    for mode in Mode:
        no_of_runs, run_times = optim.eval_run_time(mode)
        total_run_time = (
                             no_of_runs[0] * run_times[0] + no_of_runs[1] * run_times[0] + no_of_runs[2] * run_times[
                                 1]) / (
                             60 * 60)
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


def info_main():
    nn = NeuralNet()
    x_shape = cfg.DATA_SHAPE()
    test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    _, test_acc, correct = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')

    nn.net.initialize_spatial_layers(x_shape, cfg.BATCH_SIZE, PATCH_SIZE)
    nn.summary(x_shape, print_it=True)
    nn.print_weights()
    print(nn.output_size(x_shape))

    # Spatial Operations, defined one the net itself. Remember that after enabling a layer, ops are affected
    assert nn.net.num_spatial_layers() == 17
    nn.net.print_spatial_status()
    nn.train(epochs=1)  # Train to see fully disabled performance
    nn.net.print_ops_summary()
    print(nn.net.num_ops())  # (ops_saved, total_ops)

    # Given x, we generate all spatial layer requirement sizes:
    spat_sizes = nn.net.generate_spatial_sizes(x_shape)
    print(spat_sizes)
    p_spat_sizes = nn.net.generate_padded_spatial_sizes(x_shape, PATCH_SIZE)
    print(p_spat_sizes)

    # Generate a constant 1 value mask over all spatial nets
    print(nn.net.enabled_layers())
    nn.net.fill_masks_to_val(0)
    print(nn.net.enabled_layers())
    print(nn.net.disabled_layers())
    nn.net.print_spatial_status()  # Now all are enabled, seeing the mask was set
    nn.train(epochs=1)  # Train to see all layers enabled performance
    nn.net.print_ops_summary()
    nn.net.reset_spatial()  # Disables layers as well
    nn.net.print_ops_summary()
    # Turns on ids [0,3,16] and turns off all others
    nn.net.strict_mask_update(update_ids=[0, 3, 16],
                              masks=[torch.zeros(p_spat_sizes[0]), torch.zeros(p_spat_sizes[3]),
                                     torch.zeros(p_spat_sizes[16])])

    # Turns on ids [2] and *does not* turn off all others
    nn.net.lazy_mask_update(update_ids=[2], masks=[torch.zeros(p_spat_sizes[2])])
    nn.net.print_spatial_status()  # Now only 0,2,3,16 are enabled.
    print(nn.net.enabled_layers())
    nn.train(epochs=1)  # Run with 4 layers on
    nn.net.print_ops_summary()


def training_main():
    nn = NeuralNet()  # Spatial layers are by default, disabled
    nn.train(epochs=cfg.N_EPOCHS)
    test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class Optimizer:
    def __init__(self, test_gen, x_shape, patch_size, ones_range, gran_thresh, max_acc_loss):
        self.record_finder = RecordFinder(cfg.NET.__name__, patch_size, ones_range, gran_thresh, max_acc_loss)
        self.x_shape = x_shape
        self.nn = NeuralNet()
        self.test_gen = test_gen
        _, test_acc, correct = self.nn.test(self.test_gen)
        print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
        self.init_acc = 93.83  # TODO - Fix initialize bug and remove this
        self.ps = patch_size
        self.max_acc_loss = max_acc_loss
        self.gran_thresh = gran_thresh
        self.ones_range = ones_range
        self.full_net_run_time = None

    def base_line_result(self):
        layers_layout = self.nn.net.generate_spatial_sizes(self.x_shape)
        self._init_nn()

        sp_list = [None] * len(layers_layout)
        for layer, layer_mask in enumerate(mf.base_line_mask(layers_layout, self.ps)):
            sp_list[layer] = torch.from_numpy(layer_mask)
        self.nn.net.strict_mask_update(update_ids=list(range(len(layers_layout))), masks=sp_list)

        _, test_acc, _ = self.nn.test(self.test_gen)
        ops_saved, ops_total = self.nn.net.num_ops()
        bl_rec = BaselineResultRc(test_acc, ops_saved, ops_total, self.ps, cfg.NET.__name__)
        print(bl_rec)
        save_to_file(bl_rec, True, cfg.RESULTS_DIR)

    def _quantizier_main(self, rec_type, in_rec):
        if rec_type == RecordType.lQ_RESUME:
            resume_param_path = self.record_finder.find_rec_filename(in_rec.mode, RecordType.lQ_RESUME)
            quantizier = LayerQuantizier(in_rec, self.init_acc, self.ps, self.max_acc_loss, self.ones_range,
                                         resume_param_path)
        else:
            q_rec_fn = self.record_finder.find_rec_filename(in_rec.mode, rec_type)
            Quantizier = PatchQuantizier if rec_type == RecordType.pQ_REC else ChannelQuantizier
            if q_rec_fn is None:
                quantizier = Quantizier(in_rec, self.init_acc - self.max_acc_loss, self.ps)
            else:
                quantizier = Quantizier(in_rec, self.init_acc - self.max_acc_loss, self.ps, None,
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

    def eval_run_time(self, mode, no_of_tries=5):
        layers_layout = self.nn.net.generate_spatial_sizes(self.x_shape)
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
            self.nn.net.reset_ops()
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
            self.nn.net.reset_ops()
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

        layers_layout = self.nn.net.generate_spatial_sizes(self.x_shape)
        self.nn.net.reset_spatial()

        if rec_filename is None:
            rcs = Record(layers_layout, self.gran_thresh, True, mode, self.init_acc, self.ps, self.ones_range)
            st_point = [0] * 4
            rcs.filename = f'ps{self.ps}_ones{self.ones_range[0]}x{self.ones_range[1]}_{rcs.filename}'

        print('==> Result will be saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
        save_counter = 0
        for layer, channel, patch, pattern_idx, mask in tqdm(
                mf.gen_masks_with_resume(self.ps, rcs.all_patterns, rcs.mode, rcs.gran_thresh, layers_layout,
                                         resume_params=st_point)):
            self.nn.net.strict_mask_update(update_ids=[layer], masks=[torch.from_numpy(mask)])

            _, test_acc, _ = self.nn.test(self.test_gen)
            ops_saved, ops_total = self.nn.net.num_ops()
            self.nn.net.reset_ops()
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
        self.nn.net.disable_spatial_layers(list(range(len(self.nn.net.generate_spatial_sizes(CIFAR10_shape())))))
        self.nn.net.initialize_spatial_layers(CIFAR10_shape(), cfg.BATCH_SIZE, self.ps)
        _, test_acc, correct = self.nn.test(self.test_gen)
        print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
        self.nn.net.reset_ops()
        assert test_acc == self.init_acc, f'starting accuracy does not match! curr_acc:{test_acc}, prev_acc{test_acc}'


if __name__ == '__main__':
    main()
