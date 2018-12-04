# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
# Torch Libraries:
import torch
import torch.nn

# Utils:
import os
from tqdm import tqdm
from util.data_import import CIFAR10_Test,CIFAR10_shape

import Record as rc
import maskfactory as mf
from NeuralNet import NeuralNet
import Config as cfg
import RecordFinder as rf
from LayerQuantizier import LayerQuantizier
from ChannelQuantizier import ChannelQuantizier


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def info_main():
    # TODO ~!~!~!~! INNA READ ME ~!~!~!~! TODO

    nn = NeuralNet()
    x_shape = CIFAR10_shape() # (3,32,32)

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
    print(nn.net.num_ops())  # (ops_saved, total_ops)

    # Given x, we generate all spatial layer requirement sizes:
    spat_sizes = nn.net.generate_spatial_sizes(x_shape)
    print(spat_sizes)

    # Generate a constant 1 value mask over all spatial nets - equivalent to SP_ZERO with 0 and SP_ONES with 1
    # This was implemented for any constant value
    print(nn.net.enabled_layers())
    nn.net.fill_masks_to_val(1)
    print(nn.net.enabled_layers())
    print(nn.net.disabled_layers())
    nn.net.print_spatial_status()  # Now all are enabled, seeing the mask was set
    nn.train(epochs=1)  # Train to see all layers enabled performance
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

def by_uniform_layers():
    in_rec = gen_first_lvl_results_main(rc.uniform_layer)
    lQ_rec = lQ_main(in_rec)
    min_acc = rf.get_min_acc(lQ_rec.filename)
    rf.print_best_results(lQ_rec, min_acc)

def by_uniform_patches():
    in_rec = gen_first_lvl_results_main(rc.uniform_patch)
    cQ_rec = cQ_main(in_rec)
    lQ_rec = lQ_main(cQ_rec)
    min_acc = rf.get_min_acc(lQ_rec.filename)
    rf.print_best_results(lQ_rec, min_acc)

def cQ_main(in_rec):
    init_acc = rf.get_init_acc(in_rec.filename)
    cQ_rec_fn = rf.find_rec_filename(in_rec.mode, rf.cQ_REC)
    if cQ_rec_fn is None:
        cQ = ChannelQuantizier(in_rec, init_acc - cfg.MAX_ACC_LOSS, cfg.PS)
    else:
        cQ = ChannelQuantizier(in_rec, init_acc - cfg.MAX_ACC_LOSS, cfg.PS, None, \
                               rc.load_from_file(cQ_rec_fn, ''))
    cQ.simulate()
    return cQ.output_rec

def lQ_main(in_rec):
    init_acc = rf.get_init_acc(in_rec.filename)
    lQ_rec_fn = rf.find_rec_filename(in_rec.mode, rf.lQ_REC)
    if lQ_rec_fn is None:
        lQ = LayerQuantizier(in_rec, init_acc - cfg.MAX_ACC_LOSS, cfg.PS)
    else:
        lQ = LayerQuantizier(in_rec, init_acc - cfg.MAX_ACC_LOSS, cfg.PS, None, \
                             rc.load_from_file(lQ_rec_fn, ''))
    lQ.simulate()
    return lQ.output_rec

def gen_first_lvl_results_main(mode):
    rec_filename = rf.find_rec_filename(mode, rf.FIRST_LVL_REC)
    if rec_filename is not None:
        rcs = rc.load_from_file(rec_filename, path='')
        st_point = rcs.find_resume_point()
        if st_point == None:
            return rcs

    nn = NeuralNet()
    test_gen = CIFAR10_Test(batch_size=cfg.BATCH_SIZE, download=cfg.DO_DOWNLOAD, max_dataset_size=cfg.TEST_SET_SIZE)
    _, test_acc, correct = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc} [{correct}]\n ')

    # TODO ~!~!~!~! INNA READ ME ~!~!~!~! TODO
    # Remember to init spatial layers
    if rec_filename is None:  # TODO - LAYER_LAYOUT was destroyed - Need to use nn.net.generate_spatial_sizes(x_shape)
        rcs = rc.Record(cfg.LAYER_LAYOUT, cfg.GRAN_THRESH, True, mode, test_acc, cfg.PS, cfg.ONES_RANGE)
        st_point = [0] * 4
        rcs.filename = f'ps{cfg.PS}_ones{cfg.ONES_RANGE[0]}x{cfg.ONES_RANGE[1]}_{rcs.filename}'

    print('==> Result will be saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
    save_counter = 0
    for layer, channel, patch, pattern_idx, mask in tqdm(mf.gen_masks_with_resume \
                                                                     (cfg.PS, rcs.all_patterns, rcs.mode,
                                                                      rcs.gran_thresh,
                                                                      cfg.LAYER_LAYOUT, resume_params=st_point)):
        # Todo - Invalid lines
        # Mask needs to be created according to generate_spatial_sizes(x_shape)[layer] - This is O(1) operation
        sp_list = cfg.SP_MOCK
        sp_list[layer] = (1, cfg.PS, cfg.BATCH_SIZE, torch.from_numpy(mask))
        nn.update_spatial(sp_list)
        # Todo - Invalid lines
        # Correct line: 
        nn.net.strict_mask_update(update_ids=[layer], masks=[torch.from_numpy(mask)])

        _, test_acc, _ = nn.test(test_gen)
        ops_saved, ops_total = nn.net.num_ops()
        rcs.addRecord(ops_saved.item(), ops_total, test_acc, layer, channel, patch, pattern_idx)

        save_counter += 1
        if save_counter > cfg.SAVE_INTERVAL:
            rc.save_to_file(rcs, True, cfg.RESULTS_DIR)
            save_counter = 0

    rc.save_to_file(rcs, True, cfg.RESULTS_DIR)
    rcs.save_to_csv(cfg.RESULTS_DIR)
    print('==> Result saved to ' + os.path.join(cfg.RESULTS_DIR, rcs.filename))
    return rcs


if __name__ == '__main__':
    info_main()
