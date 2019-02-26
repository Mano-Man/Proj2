# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn

import glob
import matplotlib.pyplot as plt
import os

from Optimizer import Optimizer
from NeuralNet import NeuralNet
import Config as cfg
from Config import DATA as dat
import plotting
from Record import Mode, gran_dict, RecordType, load_from_file
from RecordFinder import RecordFinder
import random

# ----------------------------------------------------------------------------------------------------------------------
#                                                     Consts
# ----------------------------------------------------------------------------------------------------------------------

PATCH_SIZE = 2
RANGE_OF_ONES = (1, 3)
GRANULARITY_TH = 10
ACC_LOSS = 2
ACC_LOSS_OPTS = [0, 1, 3, 5, 10]

# ----------------------------------------------------------------------------------------------------------------------
#                                                  Generic Workloads
# ----------------------------------------------------------------------------------------------------------------------


def training(lr=0.01, epochs=50,batch_size_override=cfg.BATCH_SIZE):
    # dat.data_summary(show_sample=False)
    nn = NeuralNet(resume=True)  # Spatial layers are by default, disabled
    nn.summary(dat.shape())
    nn.train(epochs=epochs, lr=lr,batch_size=batch_size_override)
    test_gen, _ = dat.testset(batch_size=cfg.BATCH_SIZE, max_samples=cfg.TEST_SET_SIZE)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')

def test():
    nn = NeuralNet()
    test_gen, _ = dat.testset(batch_size=cfg.BATCH_SIZE, max_samples=cfg.TEST_SET_SIZE)
    nn.test(test_gen,print_it=True)

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Tutorials
# ----------------------------------------------------------------------------------------------------------------------

def info_tutorial():
    nn = NeuralNet()
    x_shape = dat.shape()
    test_gen, _ = dat.testset(batch_size=cfg.BATCH_SIZE, max_samples=cfg.TEST_SET_SIZE)
    nn.test(test_gen, print_it=True)
    nn.net.initialize_spatial_layers(x_shape, cfg.BATCH_SIZE, PATCH_SIZE)
    nn.summary(x_shape, print_it=True)
    nn.print_weights()
    print(nn.output_size(x_shape))

    # Spatial Operations, defined one the net itself. Remember that after enabling a layer, ops are affected
    assert nn.net.num_spatial_layers() != 0
    nn.net.print_spatial_status()
    # nn.train(epochs=1, set_size=5000, lr=0.1, batch_size=cfg.BATCH_SIZE)  # Train to see fully disabled performance
    nn.net.print_ops_summary()
    print(nn.net.num_ops())  # (ops_saved, total_ops)

    # Given x, we generate all spatial layer requirement sizes:
    spat_sizes = nn.net.generate_spatial_sizes(x_shape)
    print(spat_sizes)
    p_spat_sizes = nn.net.generate_padded_spatial_sizes(x_shape, PATCH_SIZE)
    print(p_spat_sizes)

    # Generate a constant 1 value mask over all spatial nets
    print(nn.net.enabled_layers())
    nn.net.fill_masks_to_val(1)
    print(nn.net.enabled_layers())
    print(nn.net.disabled_layers())
    nn.net.print_spatial_status()  # Now all are enabled, seeing the mask was set
    nn.train(epochs=1, set_size=5000, lr=0.1, batch_size=cfg.BATCH_SIZE)  # Train to see all layers enabled performance
    nn.net.print_ops_summary()
    nn.net.reset_spatial()  # Disables layers as well
    nn.net.print_ops_summary()
    # Turns on 3 ids and turns off all others
    chosen_victims = random.sample(range(nn.net.num_spatial_layers()), 4)
    nn.net.strict_mask_update(update_ids=chosen_victims[0:3],
                              masks=[torch.zeros(p_spat_sizes[chosen_victims[0]]),
                                     torch.zeros(p_spat_sizes[chosen_victims[1]]),
                                     torch.zeros(p_spat_sizes[chosen_victims[2]])])

    # Turns on one additional id and *does not* turn off all others
    nn.net.lazy_mask_update(update_ids=[chosen_victims[3]], masks=[torch.zeros(p_spat_sizes[chosen_victims[3]])])
    nn.net.print_spatial_status()  #
    print(nn.net.enabled_layers())
    nn.train(epochs=1, set_size=5000, lr=0.1, batch_size=cfg.BATCH_SIZE)  # Run with 4 layers on
    nn.net.print_ops_summary()


def data_tutorial():
    dat.data_summary()
    print(dat.name())
    print(dat.num_classes())
    print(dat.input_channels())
    print(dat.class_labels())
    print(dat.max_test_size())
    print(dat.max_train_size())
    print(dat.shape())
    (train_loader, num_train), (valid_loader, num_valid) = dat.trainset(batch_size=cfg.BATCH_SIZE,
                                                                        max_samples=dat.max_train_size(),
                                                                        show_sample=True)
    test_gen, testset_siz = dat.testset(batch_size=cfg.BATCH_SIZE, max_samples=cfg.TEST_SET_SIZE)


def debug_layer_q(acc_loss, regexes=None):
    print('debuging...')
    if regexes is None:
        regexes = [
            f'LayerQ{cfg.LQ_OPTION}_' + f'ma{acc_loss}_PatchQ_ma{acc_loss}_ResNet18Spatial_CIFAR10_acc93.5_uniform_filters_ps2_ones1x3_mg10_',
            f'LayerQ{cfg.LQ_OPTION}_' + f'ma{acc_loss}_ChannelQ_ma{acc_loss}_ResNet18Spatial_CIFAR10_acc93.5_uniform_patch_ps2_ones1x3_mg10_',
            f'LayerQ{cfg.LQ_OPTION}_' + f'ma{acc_loss}_ResNet18Spatial_CIFAR10_acc93.5_uniform_layer_ps2_ones1x3_mg10_']
    for regex in regexes:
        print(f'{cfg.RESULTS_DIR}/{regex}*pkl')
        fn = glob.glob(f'{cfg.RESULTS_DIR}/{regex}*pkl')[0]
        rec = load_from_file(fn, '')
        rec.debug(os.path.basename(fn))


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Optimization Mains
# ----------------------------------------------------------------------------------------------------------------------

def main():
    optim = Optimizer(PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH, ACC_LOSS)
    optim.base_line_result()
    optim.by_uniform_layers()
    optim.by_uniform_filters()
    optim.by_uniform_patches()
    optim.by_max_granularity()


def run_all_acc_loss_possibilities(ps, ones_range, gran_th, mode=None, acc_loss_opts=ACC_LOSS_OPTS):
    for acc_loss in acc_loss_opts:
        optim = Optimizer(ps, ones_range, gran_th, acc_loss)
        optim.run_mode(mode)


def run_all_ones_possibilities(ps, ones_possibilities, gran_th, acc_loss, mode=None):
    for ones in ones_possibilities:
        optim = Optimizer(ps, (ones, ones + 1), gran_th, acc_loss)
        optim.print_runtime_eval()
        optim.run_mode(mode)


def eval_baseline_and_runtimes(ps, ones_range, gran_th):
    optim = Optimizer(ps, ones_range, gran_th, 0)
    optim.base_line_result()
    optim.print_runtime_eval()
    return optim.init_acc


def get_init_acc(ps, ones_range, gran_th):
    optim = Optimizer(ps, ones_range, gran_th, 0)
    return optim.init_acc


def main_plot_ops_saved_vs_ones(mode):
    ps = 3
    ones_possibilities = range(3, 8)
    init_acc = get_init_acc(ps, (ones_possibilities[0], ones_possibilities[1]), GRANULARITY_TH)
    run_all_ones_possibilities(ps, ones_possibilities, GRANULARITY_TH, ACC_LOSS)
    plotting.plot_ops_saved_vs_ones(cfg.NET.__name__, dat.name(), ps, ones_possibilities,
                                    GRANULARITY_TH, ACC_LOSS, init_acc, mode)


def main_plot_ops_saved_vs_max_acc_loss(ps, ones_range, gran_th, title=None):
    init_acc = get_init_acc(ps, ones_range, gran_th)
    run_all_acc_loss_possibilities(ps, ones_range, gran_th, Mode.UNIFORM_LAYER)
    run_all_acc_loss_possibilities(ps, ones_range, gran_th, Mode.UNIFORM_FILTERS)
    run_all_acc_loss_possibilities(ps, ones_range, gran_th, Mode.UNIFORM_PATCH)
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, dat.name(), ps, ones_range,
                                            gran_th, ACC_LOSS_OPTS, init_acc, title=title)
    run_all_acc_loss_possibilities(ps, ones_range, gran_th, Mode.MAX_GRANULARITY)
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, dat.name(), ps, ones_range,
                                            gran_th, ACC_LOSS_OPTS, init_acc)


def main_ones(ones_range, acc_loss=None):
    eval_baseline_and_runtimes(2, ones_range, 10)
    if acc_loss is None:
        acc_loss = [1, 3.5, 5]
    run_all_acc_loss_possibilities(2, ones_range, 10, Mode.UNIFORM_LAYER, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(2, ones_range, 10, Mode.UNIFORM_PATCH, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(2, ones_range, 10, Mode.UNIFORM_FILTERS, acc_loss_opts=acc_loss)
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, dat.name(), 2, ones_range,
                                            10, acc_loss, 93.5)


def patch3x3():
    ones_range = (3, 4)
    gran = 32 * 32
    eval_baseline_and_runtimes(3, ones_range, gran)
    acc_loss = [9.89]
    run_all_acc_loss_possibilities(3, ones_range, gran, Mode.UNIFORM_LAYER, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(3, ones_range, gran, Mode.UNIFORM_FILTERS, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(3, ones_range, gran, Mode.UNIFORM_PATCH, acc_loss_opts=acc_loss)
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, dat.name(), 3, ones_range,
                                            gran, acc_loss, 93.5)


def main_2_ones_with_maxg():
    eval_baseline_and_runtimes(2, (2, 3), 10)
    acc_loss = [1, 3.5, 5]
    run_all_acc_loss_possibilities(2, (2, 3), 10, Mode.MAX_GRANULARITY, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(2, (2, 3), 10, Mode.UNIFORM_LAYER, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(2, (2, 3), 10, Mode.UNIFORM_PATCH, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(2, (2, 3), 10, Mode.UNIFORM_FILTERS, acc_loss_opts=acc_loss)

    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, dat.name(), 2, (2, 3),
                                            10, acc_loss, 93.5)


def main_1x3_ones():
    acc_loss = [3.5]
    run_all_acc_loss_possibilities(2, (1, 3), 10, Mode.UNIFORM_LAYER, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(2, (1, 3), 10, Mode.UNIFORM_PATCH, acc_loss_opts=acc_loss)
    run_all_acc_loss_possibilities(2, (1, 3), 10, Mode.UNIFORM_FILTERS, acc_loss_opts=acc_loss)
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, dat.name(), 2, (1, 3),
                                            10, acc_loss, 93.5)


if __name__ == '__main__':
    # training(lr=0.0001,batch_size_override=64)
    test()
    # main_ones((1,3))
    # run_all_acc_loss_possibilities(2, (2,3), 10, Mode.MAX_GRANULARITY, acc_loss_opts=[5])
