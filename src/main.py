# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn



from Optimizer import Optimizer
from NeuralNet import NeuralNet
import Config as cfg

# ----------------------------------------------------------------------------------------------------------------------
#                                                     Consts
# ----------------------------------------------------------------------------------------------------------------------

PATCH_SIZE = 3
RANGE_OF_ONES = (1, 3)
GRANULARITY_TH = 10
ACC_LOSS = 2
PATCH_SIZE = 2

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


def main():
    optim = Optimizer(PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH, ACC_LOSS)
    optim.base_line_result()
    optim.by_uniform_layers()
    optim.by_uniform_filters()
    optim.by_uniform_patches()
    optim.by_max_granularity()
    
def gen_uniform_layer_acc_loss_results(patch_size, range_ones, gran_th):
    for acc_loss in [0, 1, 2, 3, 5]:
        optim = Optimizer(patch_size, range_ones, gran_th, acc_loss)
        optim.by_uniform_layers()

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
    gen_uniform_layer_acc_loss_results(PATCH_SIZE, RANGE_OF_ONES, GRANULARITY_TH)
