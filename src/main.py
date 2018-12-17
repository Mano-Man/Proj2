# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn

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
ACC_LOSS_OPTS = [0, 1, 2, 3, 5]

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
            
def run_all_acc_loss_possibilities(ps, ones_range, gran_th, mode=None):
    for acc_loss in ACC_LOSS_OPTS:
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
    
def main_plot_ops_saved_vs_max_acc_loss():
    init_acc = eval_baseline_and_runtimes(PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH)
    run_all_acc_loss_possibilities(PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH)
    plotting.plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, cfg.DATA_NAME, PATCH_SIZE, RANGE_OF_ONES,
                                   GRANULARITY_TH, ACC_LOSS_OPTS, init_acc)

def training_main():
    nn = NeuralNet()  # Spatial layers are by default, disabled
    nn.train(epochs=cfg.N_EPOCHS)
    test_gen = cfg.TEST_GEN(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')

if __name__ == '__main__':
    main_plot_ops_saved_vs_ones()