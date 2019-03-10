import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os
import csv

from RecordFinder import RecordFinder
from Optimizer import Optimizer
from NeuralNet import NeuralNet
from util.datasets import Datasets
from Record import RecordType, Mode, gran_dict, load_from_file, save_to_file
import Config as cfg


def show_final_mask(show_all_layers=False, layers_to_show=None, show_all_channels=False,
                    channels_to_show=None, plot_3D=False, net_name='*', dataset_name='*',
                    mode=Mode.ALL_MODES, ps='*', ones_range=('*', '*'), acc_loss='*',
                    gran_thresh='*', init_acc='*'):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss, init_acc)
    final_rec_fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
    if final_rec_fn is None:
        print('No Record found')
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
            else:
                channels = channels_to_show
            for channel in channels:
                show_channel(l_to_plot_idx, channel, rec.layers_layout[l_to_plot_idx], l_to_plot[channel],
                             rec.patch_size)

    return rec

def ops_saved_summery(net_name=cfg.NET.__name__, dataset_name=cfg.DATA.name(),
                 mode=Mode.ALL_MODES, ps='*', ones_range=('*', '*'), acc_loss='*',
                 gran_thresh='*', init_acc='*', batch_size=cfg.BATCH_SIZE, 
                 max_samples=cfg.TEST_SET_SIZE):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss, init_acc)
    final_rec_fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
    if final_rec_fn is None:
        print('No Record found')
        return
    rec = load_from_file(final_rec_fn, '')
    print(rec)
    
    base_fn = 'ops_summery_'+rec.filename
    summery_fn_pkl = os.path.join(cfg.RESULTS_DIR,base_fn +'.pkl')
    if os.path.exists(summery_fn_pkl):
        arr = load_from_file(summery_fn_pkl, path='')
    else:
        nn = NeuralNet()
        data = Datasets.get(dataset_name,cfg.DATASET_DIR)
        nn.net.initialize_spatial_layers(data.shape(), cfg.BATCH_SIZE, rec.patch_size)
        test_gen, _ = data.testset(batch_size=batch_size, max_samples=max_samples)
        
        arr = [None]*len(rec.mask)
        for idx, layer in enumerate(rec.mask):
            nn.net.reset_spatial()
            print(f"----------------------------------------------------------------")
            
            nn.net.strict_mask_update(update_ids=[idx], masks=[layer])
            _, test_acc, _ = nn.test(test_gen)
            ops_saved, ops_total = nn.net.num_ops()
            
            arr[idx] = (ops_saved, ops_total, test_acc)
            nn.net.print_ops_summary()
            
        print(f"----------------------------------------------------------------")
        nn.net.reset_spatial()
        save_to_file(arr, use_default=False, path='', filename=summery_fn_pkl)
    
    out_path = os.path.join(cfg.RESULTS_DIR, base_fn + ".csv")
    with open(out_path, 'w', newline='') as f:
        csv.writer(f).writerow(
            ['layer','ops_saved', 'ops_total'])
        for idx, r in enumerate(arr):
            csv.writer(f).writerow([idx, r[0], r[1]])
    
    return arr
    

def show_channel_grid(layer=0, net_name='*', dataset_name='*',
                 mode=Mode.ALL_MODES, ps='*', ones_range=('*', '*'), acc_loss='*',
                 gran_thresh='*', init_acc='*', filename=None):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss, init_acc)
    final_rec_fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
    if final_rec_fn is None:
        print('No Record found')
        return
    rec = load_from_file(final_rec_fn, '')
    
    layer_mask = rec.mask[layer].numpy()
    
    no_of_channels = layer_mask.shape[0]
    rows = math.ceil(math.sqrt(no_of_channels))
    fig, axs = plt.subplots(nrows=rows, ncols=rows)
    fig.set_figheight(30) 
    fig.set_figwidth(30)
    
    for c in range(layer_mask.shape[0]):
        row_idx = math.floor(c/rows)
        col_idx = c - row_idx*rows
        axs[row_idx][col_idx].imshow(layer_mask[c], cmap=plt.cm.gray) #(0:black, 1:white)
        axs[row_idx][col_idx].set_title(f'Channel:{c}')
        # Minor ticks
        axs[row_idx][col_idx].set_xticks(np.arange(-.5, layer_mask[c].shape[0] - 1, rec.patch_size), minor=True);
        axs[row_idx][col_idx].set_yticks(np.arange(-.5, layer_mask[c].shape[1] - 1, rec.patch_size), minor=True);
        # Gridlines based on minor ticks
        #axs[row_idx][col_idx].grid(which='minor', color='r', linestyle='-', linewidth=2)
        axs[row_idx][col_idx].tick_params(axis='both', which='major', bottom=False, top=False,
                        left=False, right=False, labelbottom=False, labelleft=False)
        #axs[row_idx][col_idx].colorbar()
    plt.tight_layout()    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return rec
    

def show_final_mask_simplegrid_resnet18(
                    channel=0, net_name='*', dataset_name='*',
                    mode=Mode.ALL_MODES, ps='*', ones_range=('*', '*'), acc_loss='*',
                    gran_thresh='*', init_acc='*', filename=None):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss, init_acc)
    final_rec_fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
    if final_rec_fn is None:
        print('No Record found')
        return
    rec = load_from_file(final_rec_fn, '')
    
    grid = (4,5)
    st = ((0,0), (0,1), (0,2), (0,3), (0,4),
          (1,0), (1,1), (1,2), (1,3),
          (2,0), (2,1), (2,2), (2,3),
          (3,0), (3,1), (3,2), (3,3))
    
    fig = plt.figure()
    fig.set_figheight(10) 
    fig.set_figwidth(14)
    for l_to_plot_idx, l_to_plot in enumerate(rec.mask):
        l_to_plot = l_to_plot.numpy()
        
        
        plt.subplot2grid(grid, st[l_to_plot_idx])
        plt.imshow(l_to_plot[channel], cmap=plt.cm.gray) #(0:black, 1:white)
        plt.title(f'Layer:{l_to_plot_idx}')
        ax = plt.gca()
        # Minor ticks
        ax.set_xticks(np.arange(-.5, l_to_plot[channel].shape[0] - 1, rec.patch_size), minor=True);
        ax.set_yticks(np.arange(-.5, l_to_plot[channel].shape[1] - 1, rec.patch_size), minor=True);
        # Gridlines based on minor ticks
        #ax.grid(which='minor', color='r', linestyle='-', linewidth=2)
        plt.tick_params(axis='both', which='major', bottom=False, top=False,
                        left=False, right=False, labelbottom=False, labelleft=False)
        #plt.colorbar()
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

def show_final_mask_grid_resnet18(
                    channel=0, net_name='*', dataset_name='*',
                    mode=Mode.ALL_MODES, ps='*', ones_range=('*', '*'), acc_loss='*',
                    gran_thresh='*', init_acc='*', filename=None):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss, init_acc)
    final_rec_fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
    if final_rec_fn is None:
        print('No Record found')
        return
    rec = load_from_file(final_rec_fn, '')
    
    shift = 4
    grid = (18,24)
    st = ((0,0), (0,8), (0,16), (8,0), (8,8),
          (8,16), (8,20), (12,16), (12,20),
          (16,0+shift), (16,2+shift), (16,4+shift), (16,6+shift), (16,8+shift), (16,10+shift), (16,12+shift), (16, 14+shift))
    span = (8,8,8,8,8,4,4,4,4,2,2,2,2,2,2,2,2)
    
    fig = plt.figure()
    fig.set_figheight(10) 
    fig.set_figwidth(15)
    for l_to_plot_idx, l_to_plot in enumerate(rec.mask):
        l_to_plot = l_to_plot.numpy()
        
        
        plt.subplot2grid(grid, st[l_to_plot_idx], colspan=span[l_to_plot_idx], rowspan=span[l_to_plot_idx])
        plt.imshow(l_to_plot[channel], cmap=plt.cm.gray) #(0:black, 1:white)
        plt.title(f'Layer:{l_to_plot_idx}')
        ax = plt.gca()
        # Minor ticks
        ax.set_xticks(np.arange(-.5, l_to_plot[channel].shape[0] - 1, rec.patch_size), minor=True);
        ax.set_yticks(np.arange(-.5, l_to_plot[channel].shape[1] - 1, rec.patch_size), minor=True);
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='r', linestyle='-', linewidth=2)
        plt.tick_params(axis='both', which='major', bottom=False, top=False,
                        left=False, right=False, labelbottom=False, labelleft=False)
    plt.tight_layout()    
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

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
    plt.savefig(f'{cfg.RESULTS_DIR}/ops_saved_vs_number_of_ones_{net_name}_{dataset_name}' +
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
              f'PATCH SIZE:{ps}, ONES:{ones_range[0]}-{ones_range[1]-1}, GRANULARITY:{gran_thresh}\n'
              f'LQ{cfg.LQ_OPTION}, CQ{cfg.CQ_OPTION}r{cfg.CHANNELQ_UPDATE_RATIO}, PQ{cfg.PQ_OPTION}r{cfg.PATCHQ_UPDATE_RATIO}')

    plt.legend()
    # plt.show()
    plt.savefig(f'{cfg.RESULTS_DIR}/ops_saved_vs_max_acc_loss_{net_name}_{dataset_name}_acc{init_acc}_' +
                f'LQ{cfg.LQ_OPTION}_CQ{cfg.CQ_OPTION}r{cfg.CHANNELQ_UPDATE_RATIO}_PQ{cfg.PQ_OPTION}r{cfg.PATCHQ_UPDATE_RATIO}_' +
                f'ps{ps}_ones{ones_range[0]}x{ones_range[1]}_mg{gran_thresh}.pdf')


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


if __name__ == '__main__':
    
    arr_layer = ops_saved_summery(net_name=cfg.NETS[0].__name__, dataset_name='CIFAR10',
                 mode=Mode.UNIFORM_LAYER, ps=2, ones_range=(1, 3), acc_loss=3.5,
                 gran_thresh=10, init_acc=93.5, batch_size=cfg.BATCH_SIZE, 
                 max_samples=cfg.TEST_SET_SIZE)
    
    arr_patch = ops_saved_summery(net_name=cfg.NETS[0].__name__, dataset_name='CIFAR10',
                 mode=Mode.UNIFORM_PATCH, ps=2, ones_range=(1, 3), acc_loss=3.5,
                 gran_thresh=10, init_acc=93.5, batch_size=cfg.BATCH_SIZE, 
                 max_samples=cfg.TEST_SET_SIZE)
    
    arr_filters = ops_saved_summery(net_name=cfg.NETS[0].__name__, dataset_name='CIFAR10',
                 mode=Mode.UNIFORM_FILTERS, ps=2, ones_range=(1, 3), acc_loss=3.5,
                 gran_thresh=10, init_acc=93.5, batch_size=cfg.BATCH_SIZE, 
                 max_samples=cfg.TEST_SET_SIZE)
    
    arr_gran = ops_saved_summery(net_name=cfg.NETS[0].__name__, dataset_name='CIFAR10',
                 mode=Mode.MAX_GRANULARITY, ps=2, ones_range=(1, 3), acc_loss=3.5,
                 gran_thresh=10, init_acc=93.5, batch_size=cfg.BATCH_SIZE, 
                 max_samples=cfg.TEST_SET_SIZE)
    
#    show_final_mask_grid_resnet18(
#                    channel=2, net_name=cfg.NETS[0].__name__, dataset_name=cfg.DATA.name(),
#                    mode=Mode.UNIFORM_PATCH, ps=2, ones_range=(1, 3), acc_loss=3.5,
#                    gran_thresh=10, init_acc=93.5)
#    
#    show_final_mask_simplegrid_resnet18(
#                    channel=2, net_name=cfg.NETS[0].__name__, dataset_name=cfg.DATA.name(),
#                    mode=Mode.UNIFORM_PATCH, ps=2, ones_range=(1, 3), acc_loss=3.5,
#                    gran_thresh=10, init_acc=93.5)
    
#    rec = show_channel_grid(layer=4, net_name=cfg.NETS[0].__name__, dataset_name=cfg.DATA.name(),
#                 mode=Mode.UNIFORM_PATCH, ps=2, ones_range=(1, 3), acc_loss=3.5,
#                 gran_thresh=10, init_acc=93.5)
#    
#    show_final_mask(show_all_layers=True, layers_to_show=None, show_all_channels=False,
#                    channels_to_show=None, plot_3D=False, net_name=cfg.NETS[0].__name__, dataset_name=cfg.DATA.name(),
#                    mode=Mode.UNIFORM_FILTERS, ps=2, ones_range=(1, 3), acc_loss=3.5,
#                    gran_thresh=10, init_acc=93.5)
    
#    rec = show_final_mask(show_all_layers=False, layers_to_show=[], show_all_channels=False,
#                    channels_to_show=None, plot_3D=False, net_name=cfg.NETS[0].__name__, dataset_name=cfg.DATA.name(),
#                    mode=Mode.UNIFORM_PATCH, ps=2, ones_range=(1, 3), acc_loss=3.5,
#                    gran_thresh=10, init_acc=93.5)
#    uniform_patch_mask = [None]*len(rec.mask)
#    for idx, l_mask in enumerate(rec.mask):
#        uniform_patch_mask[idx] = l_mask.numpy()
#        
#    rec = show_final_mask(show_all_layers=False, layers_to_show=[], show_all_channels=False,
#                    channels_to_show=None, plot_3D=False, net_name=cfg.NETS[0].__name__, dataset_name=cfg.DATA.name(),
#                    mode=Mode.MAX_GRANULARITY, ps=2, ones_range=(1, 3), acc_loss=3.5,
#                    gran_thresh=10, init_acc=93.5)
#    max_garn_mask = [None]*len(rec.mask)
#    for idx, l_mask in enumerate(rec.mask):
#        max_garn_mask[idx] = l_mask.numpy()