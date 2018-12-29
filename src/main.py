# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn

import glob
import matplotlib.pyplot as plt

from Optimizer import Optimizer
from NeuralNet import NeuralNet
import Config as cfg
import plotting
from Record import Mode, gran_dict, RecordType, load_from_file
from RecordFinder import RecordFinder

# ----------------------------------------------------------------------------------------------------------------------
#                                                     Consts
# ----------------------------------------------------------------------------------------------------------------------

PATCH_SIZE = 2
RANGE_OF_ONES = (1, 3)
GRANULARITY_TH = 10
ACC_LOSS = 2
ACC_LOSS_OPTS = [0, 1, 3, 5, 10]

# ----------------------------------------------------------------------------------------------------------------------
#                                                    
# ----------------------------------------------------------------------------------------------------------------------

def debug_main():
    nn = NeuralNet()
    test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD, max_dataset_size=cfg.TEST_SET_SIZE)
    _, test_acc, correct = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
    nn.net.initialize_spatial_layers(cfg.DATA_SHAPE(), cfg.BATCH_SIZE, PATCH_SIZE)

    nn.net.fill_masks_to_val(0)
    test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD, max_dataset_size=cfg.TEST_SET_SIZE)
    _, test_acc, correct = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
    nn.net.print_ops_summary()
            
def run_all_acc_loss_possibilities(ps, ones_range, gran_th, mode=None, acc_loss_opts=ACC_LOSS_OPTS):
    for acc_loss in acc_loss_opts:
        optim = Optimizer(ps, ones_range, gran_th, acc_loss)
        optim.run_mode(mode)
        
def run_all_ones_possibilities(ps, ones_possibilities, gran_th, acc_loss, mode=None):
    for ones in ones_possibilities:
        optim = Optimizer(ps, (ones,ones+1), gran_th, acc_loss)
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
    ones_possibilities = range(3,8)
    init_acc = get_init_acc(ps, (ones_possibilities[0],ones_possibilities[1]), GRANULARITY_TH)
    run_all_ones_possibilities(ps, ones_possibilities, GRANULARITY_TH, ACC_LOSS)
    plotting.plot_ops_saved_vs_ones(cfg.NET.__name__, cfg.DATA_NAME, ps, ones_possibilities, 
                           GRANULARITY_TH, ACC_LOSS, init_acc, mode)
    
def main_plot_ops_saved_vs_max_acc_loss(ps, ones_range, gran_th, title=None):
    init_acc = get_init_acc(ps, ones_range, gran_th)
    run_all_acc_loss_possibilities(ps, ones_range, gran_th, Mode.UNIFORM_LAYER)
    run_all_acc_loss_possibilities(ps, ones_range, gran_th, Mode.UNIFORM_FILTERS)
    run_all_acc_loss_possibilities(ps, ones_range, gran_th, Mode.UNIFORM_PATCH)
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, cfg.DATA_NAME, ps, ones_range,
                                   gran_th, ACC_LOSS_OPTS, init_acc, title=title)
    run_all_acc_loss_possibilities(ps, ones_range, gran_th, Mode.MAX_GRANULARITY)
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, cfg.DATA_NAME, ps, ones_range,
                                   gran_th, ACC_LOSS_OPTS, init_acc)

def training_main():
    nn = NeuralNet()  # Spatial layers are by default, disabled
    nn.train(epochs=cfg.N_EPOCHS)
    test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')
    
def debug():
    print('debuging...')
    regexes = ['LayerQ_ma3.5_PatchQ_ma3.5_ResNet18Spatial_CIFAR10_acc93.5_uniform_filters_ps2_ones1x3_mg10_',
               'LayerQ_ma3.5_ChannelQ_ma3.5_ResNet18Spatial_CIFAR10_acc93.5_uniform_patch_ps2_ones1x3_mg10_',
               'LayerQ_ma3.5_ResNet18Spatial_CIFAR10_acc93.5_uniform_layer_ps2_ones1x3_mg10_']
    for regex in regexes:
        fn = glob.glob(f'{cfg.RESULTS_DIR}{regex}*pkl')[0]
        rec = load_from_file(fn, '')
        plt.figure()
        plt.subplot(221)
        plt.plot(list(range(len(rec.test_acc_array))), rec.test_acc_array,'o--') 
        plt.ylabel('acc [%]') 
    
        plt.subplot(222)
        plt.plot(list(range(len(rec.ops_saved_array))), rec.ops_saved_array,'o--') 
        plt.ylabel('ops saved') 
        print('debuging...')
        plt.subplot(223)
        plt.plot(list(range(len(rec.acc_diff_arr))), rec.acc_diff_arr,'o--') 
        plt.ylabel('acc diff')
        
        plt.subplot(224)
        plt.plot(list(range(len(rec.ops_diff_arr))), rec.ops_diff_arr,'o--') 
        plt.ylabel('ops diff')
        print('debuging...')
        plt.savefig(f'{cfg.RESULTS_DIR}debug_{regex}.pdf')

if __name__ == '__main__':
    #eval_baseline_and_runtimes(2,(1,3),10)
    run_all_acc_loss_possibilities(2, (1,3), 10, Mode.UNIFORM_LAYER, acc_loss_opts=[3.5])
    run_all_acc_loss_possibilities(2, (1,3), 10, Mode.UNIFORM_PATCH, acc_loss_opts=[3.5])
    run_all_acc_loss_possibilities(2, (1,3), 10, Mode.UNIFORM_FILTERS, acc_loss_opts=[3.5])
    debug()
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, cfg.DATA_NAME, 2, (1,3),
                                   10, [0, 1, 2, 3, 3.5, 5, 10], 93.5)
    #main_plot_ops_saved_vs_ones(Mode.UNIFORM_LAYER)