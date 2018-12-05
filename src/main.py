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
import RecordFinder as rf
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


# ----------------------------------------------------------------------------------------------------------------------
#                                         To be migrated to Optimizer
# ----------------------------------------------------------------------------------------------------------------------

def quantizier_main(Quantizier, in_rec, rec_type):
    init_acc = rf.get_init_acc(in_rec.filename)
    q_rec_fn = rf.find_rec_filename(in_rec.mode, rec_type)
    if q_rec_fn is None or rf.lQ_RESUME == rec_type:
        quantizier = Quantizier(in_rec, init_acc - cfg.MAX_ACC_LOSS, cfg.PS)
    else:
        quantizier = Quantizier(in_rec, init_acc - cfg.MAX_ACC_LOSS, cfg.PS, None, \
                                load_from_file(q_rec_fn, ''))
    quantizier.simulate()
    if  rf.lQ_RESUME == rec_type:
        return
    return quantizier.output_rec


def by_uniform_layers():
    in_rec = gen_first_lvl_results_main(Mode.UNIFORM_LAYER)
    quantizier_main(LayerQuantizier, in_rec, rf.lQ_RESUME)
    rf.print_result(Mode.UNIFORM_LAYER)


def by_uniform_patches():
    in_rec = gen_first_lvl_results_main(Mode.UNIFORM_PATCH)
    cQ_rec = quantizier_main(ChannelQuantizier, in_rec, rf.cQ_REC)
    quantizier_main(LayerQuantizier, cQ_rec, rf.lQ_RESUME)
    rf.print_result(Mode.UNIFORM_PATCH)


def by_uniform_filters():
    in_rec = gen_first_lvl_results_main(Mode.UNIFORM_FILTERS)
    pQ_rec = quantizier_main(PatchQuantizier, in_rec, rf.pQ_REC)
    quantizier_main(LayerQuantizier, pQ_rec, rf.lQ_RESUME)
    rf.print_result(Mode.UNIFORM_FILTERS)


def by_max_granularity():
    in_rec = gen_first_lvl_results_main(Mode.MAX_GRANULARITY)
    pQ_rec = quantizier_main(PatchQuantizier, in_rec, rf.pQ_REC)
    cQ_rec = quantizier_main(ChannelQuantizier, pQ_rec, rf.cQ_REC)
    quantizier_main(LayerQuantizier, cQ_rec, rf.lQ_RESUME)
    rf.print_result(Mode.MAX_GRANULARITY)


def gen_first_lvl_results_main(mode):
    rec_filename = rf.find_rec_filename(mode, rf.FIRST_LVL_REC)
    if rec_filename is not None:
        rcs = load_from_file(rec_filename, path='')
        st_point = rcs.find_resume_point()
        if st_point == None:
            return rcs

    nn = NeuralNet()
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD, max_dataset_size=cfg.TEST_SET_SIZE)
    _, test_acc, correct = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
    nn.net.initialize_spatial_layers(CIFAR10_shape(), cfg.BATCH_SIZE, cfg.PS)
    print('gen another test')
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD, max_dataset_size=cfg.TEST_SET_SIZE)
    _, test_acc, correct = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')
    layers_layout = nn.net.generate_spatial_sizes(CIFAR10_shape())


    if rec_filename is None:  
        rcs = Record(layers_layout, cfg.GRAN_THRESH, True, mode, test_acc, cfg.PS, cfg.ONES_RANGE)
        st_point = [0] * 4
        rcs.filename = f'ps{cfg.PS}_ones{cfg.ONES_RANGE[0]}x{cfg.ONES_RANGE[1]}_{rcs.filename}'

    print('==> Result will be saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
    save_counter = 0
    for layer, channel, patch, pattern_idx, mask in tqdm(mf.gen_masks_with_resume \
                                                                     (cfg.PS, rcs.all_patterns, rcs.mode,
                                                                      rcs.gran_thresh,
                                                                      layers_layout, resume_params=st_point)):
        nn.net.strict_mask_update(update_ids=[layer], masks=[torch.from_numpy(mask)])

        _, test_acc, _ = nn.test(test_gen)
        ops_saved, ops_total = nn.net.num_ops()
        rcs.addRecord(ops_saved, ops_total, test_acc, layer, channel, patch, pattern_idx)

        save_counter += 1
        if save_counter > cfg.SAVE_INTERVAL:
            save_to_file(rcs, True, cfg.RESULTS_DIR)
            save_counter = 0

    save_to_file(rcs, True, cfg.RESULTS_DIR)
    rcs.save_to_csv(cfg.RESULTS_DIR)
    print('==> Result saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
    return rcs


if __name__ == '__main__':
    by_uniform_layers()
    by_uniform_patches()
    by_uniform_filters()
