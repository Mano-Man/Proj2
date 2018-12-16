# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn

import matplotlib.pyplot as plt

from Optimizer import Optimizer
from NeuralNet import NeuralNet
import Config as cfg
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
        
def eval_baseline_and_runtimes(ps, ones_range, gran_th):
    optim = Optimizer(ps, ones_range, gran_th, 0)
    optim.base_line_result()
    optim.print_runtime_eval()
    
def main():
    run_all_acc_loss_possibilities(PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH)
    plot_ops_saved_vs_max_acc_loss(cfg.NET.__name__, cfg.DATA_NAME, PATCH_SIZE, RANGE_OF_ONES,
                                   GRANULARITY_TH, 93.5)
    
def plot_ops_saved_vs_max_acc_loss(net_name, dataset_name, ps, ones_range, gran_thresh, init_acc, mode=None):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, '*', init_acc)
    bs_line_fn = rec_finder.find_rec_filename(mode, RecordType.BASELINE_REC)
    if bs_line_fn is None:
        optim = Optimizer(ps, ones_range, gran_thresh, 0, init_acc)
        optim.base_line_result()
        bs_line_fn = rec_finder.find_rec_filename(mode, RecordType.BASELINE_REC)
    bs_line_rec = load_from_file(bs_line_fn, '')
    plt.figure()
    plt.plot(ACC_LOSS_OPTS, [round(bs_line_rec.ops_saved/bs_line_rec.total_ops, 3)]*len(ACC_LOSS_OPTS),'o--', label='baseline')
    if mode is None:
        modes = [m for m in Mode]
    else:
        modes = [mode]
        
    for mode in modes:
        fns = rec_finder.find_all_FRs(mode)
        max_acc_loss = [None]*len(fns)
        ops_saved = [None]*len(fns)
        for idx, fn in enumerate(fns):
            final_rec = load_from_file(fn,'')
            ops_saved[idx] = round(final_rec.ops_saved/final_rec.total_ops, 3)
            max_acc_loss[idx] = final_rec.max_acc_loss
        if len(fns)!= 0:    
            plt.plot(max_acc_loss, ops_saved,'o--', label=gran_dict[mode])
    
    plt.xlabel('max acc loss [%]') 
    plt.ylabel('operations saved [%]') 

    plt.title(f'Operations Saved vs Maximun Allowed Accuracy Loss \n'
              f'{net_name}, {dataset_name}, INITIAL ACC:{init_acc} \n'
              f'PATCH SIZE:{ps}, ONES:{ones_range[0]}-{ones_range[1]-1}, GRANULARITY:{gran_thresh}')

    plt.legend() 
    #plt.show() 
    plt.savefig(f'{cfg.RESULTS_DIR}ops_saved_vs_max_acc_loss_{net_name}_{dataset_name}'+
                f'acc{init_acc}_ps{ps}_ones{ones_range[0]}x{ones_range[1]}_mg{gran_thresh}.pdf')

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




if __name__ == '__main__':
    main()