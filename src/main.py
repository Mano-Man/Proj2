# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
# Torch Libraries:
import torch
import torch.nn

# Utils:
import os
from tqdm import tqdm
from util.data_import import CIFAR10_Test, CIFAR10_shape

from Record import Mode, Record, load_from_file, save_to_file
import maskfactory as mf
from NeuralNet import NeuralNet
import Config as cfg
from RecordFinder import RecordFinder, RecordType
from LayerQuantizier import LayerQuantizier
from ChannelQuantizier import ChannelQuantizier
from PatchQuantizier import PatchQuantizier


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def info_main():
    nn = NeuralNet()
    x_shape = CIFAR10_shape()  # (3,32,32)

    # Global info - From Net Wrapper
    nn.summary(x_shape, print_it=True)
    nn.print_weights()

    # Spatial Operations, defined one the net itself. Remember that after enabling a layer, ops are affected
    assert nn.net.num_spatial_layers() == 17

    nn.net.print_spatial_status()
    nn.net.initialize_spatial_layers(x_shape, cfg.BATCH_SIZE, cfg.PS)  # Must be done if we want to use the spat layers
    nn.net.print_spatial_status()
    nn.train(epochs=1)  # Train to see disabled performance
    nn.net.print_ops_summary()
    #print(nn.net.num_ops())  # (ops_saved, total_ops)

    # Given x, we generate all spatial layer requirement sizes:
    spat_sizes = nn.net.generate_spatial_sizes(x_shape)
    print(spat_sizes)

    # Generate a constant 1 value mask over all spatial nets - equivalent to SP_ZERO with 0 and SP_ONES with 1
    # This was implemented for any constant value
    print(nn.net.enabled_layers())
    nn.net.fill_masks_to_val(0)
    print(nn.net.enabled_layers())
    print(nn.net.disabled_layers())
    nn.net.print_spatial_status()  # Now all are enabled, seeing the mask was set

    nn.train(epochs=1)  # Train to see all layers enabled performance
    nn.net.print_ops_summary()
    nn.net.reset_ops()
    nn.net.print_ops_summary()

    # Turns on ids [0,3,16] and turns off all others
    nn.net.strict_mask_update(update_ids=[0, 3, 16],
                              masks=[torch.zeros(spat_sizes[0]), torch.zeros(spat_sizes[3]),
                                     torch.zeros(spat_sizes[16])])

    # Turns on ids [2] and *does not* turn off all others
    nn.net.lazy_mask_update(update_ids=[2], masks=[torch.zeros(spat_sizes[2])])
    nn.net.print_spatial_status()  # Now only 0,2,3,16 are enabled.
    print(nn.net.enabled_layers())
    nn.train(epochs=1)  # Run with 4 layers on
    nn.net.print_ops_summary()


def training_main():
    nn = NeuralNet()  # Spatial layers are by default, disabled
    nn.train(epochs=cfg.N_EPOCHS)
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')


if __name__ == '__main__':
    info_main()
