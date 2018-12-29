import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from RecordFinder import RecordFinder
from Optimizer import Optimizer
from Record import RecordType, Mode, gran_dict, load_from_file
import Config as cfg


def show_final_mask(show_all_layers=False, layers_to_show=None, show_all_channels=False,
                    channels_to_show=None, plot_3D=False, net_name='*', dataset_name='*',
                    mode=Mode.ALL_MODES, ps='*', ones_range=('*', '*'), acc_loss='*',
                    gran_thresh='*', init_acc='*'):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss, init_acc)
    final_rec_fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
    if final_rec_fn is None:
        return
    rec = load_from_file(final_rec_fn, '')
    print(rec)
    mask_size = [rec.layers_layout[l][0] * rec.layers_layout[l][1] * rec.layers_layout[l][2] for l in
                 range(len(rec.mask))]
    zeros_in_each_layer = [np.count_nonzero(rec.mask[l].numpy() == 0) / mask_size[l] for l in range(len(rec.mask))]

    plt.figure()
    tick_label = [str(l) for l in range(len(rec.mask))]
    plt.bar(list(range(len(rec.mask))), zeros_in_each_layer, tick_label=tick_label)
    plt.xlabel('layer index')
    plt.ylabel('zeros [%]')
    plt.title('[%] of Zeros in each Prediction Layer for the Chosen Mask')
    plt.show()

    if show_all_layers:
        layers_to_show = range(len(rec.mask))
    elif layers_to_show is None:
        layers_to_show = [max(range(len(zeros_in_each_layer)), key=zeros_in_each_layer.__getitem__)]

    for idx, l_to_plot_idx in enumerate(layers_to_show):
        l_to_plot = rec.mask[l_to_plot_idx].numpy()
        if rec.mode == Mode.UNIFORM_FILTERS or rec.mode == Mode.UNIFORM_LAYER:  # all channels in layer are the same
            show_channel(l_to_plot_idx, 0, rec.layers_layout[l_to_plot_idx], l_to_plot[0], rec.patch_size)
        else:
            if plot_3D:
                show_layer(l_to_plot_idx, rec.layers_layout[l_to_plot_idx], l_to_plot)
            if show_all_channels:
                channels = range(rec.layers_layout[l_to_plot_idx][0])
            elif channels_to_show is None:
                channels = [0, round(rec.layers_layout[l_to_plot_idx][0] / 2), rec.layers_layout[l_to_plot_idx][0] - 1]
            elif type(channels_to_show) is list and type(channels_to_show[0]) is list:
                channels = channels_to_show[idx]
            elif type(channels_to_show) is not list:
                channels = [channels_to_show]
            for channel in channels:
                show_channel(l_to_plot_idx, channel, rec.layers_layout[l_to_plot_idx], l_to_plot[channel],
                             rec.patch_size)

    return rec


def plot_ops_saved_vs_ones(net_name, dataset_name, ps, ones_possibilities, gran_thresh, acc_loss, init_acc,
                           modes=None):
#    bs_line_rec = get_baseline_rec(net_name, dataset_name, ps, init_acc)
    plt.figure()
#    if bs_line_rec is not None:
#        plt.plot(ones_possibilities, [bs_line_rec.ops_saved/bs_line_rec.total_ops]*len(ones_possibilities),
#                                      '--', label=f'baseline, {round(bs_line_rec.init_acc-bs_line_rec.baseline_acc, 2)}% loss')
    modes = get_modes(modes)
    for mode in modes:
        ops_saved = [None] * len(ones_possibilities)
        has_results = False
        for idx, ones in enumerate(ones_possibilities):
            rec_finder = RecordFinder(net_name, dataset_name, ps, (ones, ones + 1), gran_thresh, acc_loss, init_acc)
            fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
            if fn is not None:
                rec = load_from_file(fn, '')
                ops_saved[idx] = round(rec.ops_saved / rec.total_ops, 3)
                has_results = True
        if has_results:
            plt.plot(ones_possibilities, ops_saved, 'o--', label=gran_dict[mode])
    plt.xlabel('number of ones')
    plt.ylabel('operations saved [%]')
    plt.title(f'Operations Saved vs Number of Ones \n'
              f'{net_name}, {dataset_name}, INITIAL ACC:{init_acc} \n'
              f'PATCH SIZE:{ps}, MAX ACC LOSS:{acc_loss}, GRANULARITY:{gran_thresh}')
    plt.legend()
    plt.savefig(f'{cfg.RESULTS_DIR}ops_saved_vs_number_of_ones_{net_name}_{dataset_name}' +
                f'acc{init_acc}_ps{ps}_ma{acc_loss}_mg{gran_thresh}.pdf')


def plot_ops_saved_vs_max_acc_loss(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss_opts, init_acc,
                                   modes=None, title=None):
    bs_line_rec = get_baseline_rec(net_name, dataset_name, ps, init_acc)
    plt.figure()
    if bs_line_rec is not None:
        plt.plot(acc_loss_opts, [round(bs_line_rec.ops_saved / bs_line_rec.total_ops, 3)] * len(acc_loss_opts),
                 '--', label=f'baseline')
        plt.axvline(x=bs_line_rec.init_acc - bs_line_rec.baseline_acc, linestyle='--', label='baseline')

    modes = get_modes(modes)
    for mode in modes:
        ops_saved = [None] * len(acc_loss_opts)
        for idx, acc_loss in enumerate(acc_loss_opts):
            rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss, init_acc)
            fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
            if fn is not None:
                final_rec = load_from_file(fn, '')
                ops_saved[idx] = round(final_rec.ops_saved / final_rec.total_ops, 3)
        if (ops_saved != [None] * len(acc_loss_opts)):
            plt.plot(acc_loss_opts, ops_saved, 'o--', label=gran_dict[mode])

    plt.xlabel('max acc loss [%]')
    plt.ylabel('operations saved [%]')

    if title is None:
        title = ''
    plt.title(f'Operations Saved vs Maximun Allowed Accuracy Loss {title}\n'
              f'{net_name}, {dataset_name}, INITIAL ACC:{init_acc} \n'
              f'PATCH SIZE:{ps}, ONES:{ones_range[0]}-{ones_range[1]-1}, GRANULARITY:{gran_thresh}')

    plt.legend()
    # plt.show()
    plt.savefig(f'{cfg.RESULTS_DIR}ops_saved_vs_max_acc_loss_{net_name}_{dataset_name}' +
                f'acc{init_acc}_ps{ps}_ones{ones_range[0]}x{ones_range[1]}_mg{gran_thresh}.pdf')


def show_channel(layer, channel, dims, image, ps, filename=None):
    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray) #(0:black, 1:white)
    plt.title(f'Layer:{layer}, Channel:{channel}, dims:{dims}')
    ax = plt.gca()
    # Minor ticks
    ax.set_xticks(np.arange(-.5, image.shape[0] - 1, ps), minor=True);
    ax.set_yticks(np.arange(-.5, image.shape[1] - 1, ps), minor=True);
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='r', linestyle='-', linewidth=2)
    plt.tick_params(axis='both', which='major', bottom=False, top=False,
                    left=False, right=False, labelbottom=False, labelleft=False)
    plt.colorbar()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def show_layer(layer_idx, dims, layer, filename=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("Channels")
    ax.set_ylabel("N")
    ax.set_zlabel("M")
    ax.grid(True)
    ax.voxels(layer, edgecolors='gray')
    plt.title(f'Layer:{layer_idx}, dims:{dims}')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


def get_modes(modes):
    if modes is None:
        modes = [m for m in Mode if m is not Mode.ALL_MODES]
    elif type(modes) == list:
        modes = modes
    else:
        modes = [modes]
    return modes


def get_baseline_rec(net_name, dataset_name, ps, init_acc):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ('*', '*'), '*', '*', init_acc)
    bs_line_fn = rec_finder.find_rec_filename(None, RecordType.BASELINE_REC)
    if bs_line_fn is None:
        optim = Optimizer(ps, (None, None), None, None)
        optim.base_line_result()
        bs_line_fn = rec_finder.find_rec_filename(None, RecordType.BASELINE_REC)
    if bs_line_fn is None:
        print(f' !!! Was not able to get baseline result for initial accuracy of {init_acc} !!!')
        print(f' !!! Adjust TEST_SET_SIZE in Config.py !!!')
        return bs_line_fn
    return load_from_file(bs_line_fn, '')
