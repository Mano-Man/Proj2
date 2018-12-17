import numpy as np
import matplotlib.pyplot as plt

from RecordFinder import RecordFinder
from Optimizer import Optimizer
from Record import RecordType, Mode, gran_dict, load_from_file
import Config as cfg
    
def show_final_mask(net_name, dataset_name, mode, show_all_layers=False, ps='*', ones_range=('*','*'), gran_thresh='*', init_acc='*'):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, '*', init_acc)
    final_rec_fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
    if final_rec_fn is None:
        return
    rec = load_from_file(final_rec_fn, '')
    print(rec)
    mask_size = [rec.layers_layout[l][0]*rec.layers_layout[l][1]*rec.layers_layout[l][2] for l in range(len(rec.mask))]
    zeros_in_each_layer = [np.count_nonzero(rec.mask[l].numpy()==0)/mask_size[l] for l in range(len(rec.mask))]
    
    plt.figure()
    tick_label = [str(l) for l in range(len(rec.mask))]
    plt.bar(list(range(len(rec.mask))), zeros_in_each_layer, tick_label = tick_label) 
    plt.xlabel('layer index') 
    plt.ylabel('zeros [%]')  
    plt.show()
    
    if show_all_layers:
        indexes = range(len(rec.mask))
    else:
        indexes = [max(range(len(zeros_in_each_layer)), key=zeros_in_each_layer.__getitem__)]
    for l_to_plot_idx in indexes:
        l_to_plot = rec.mask[l_to_plot_idx].numpy()
        if mode==Mode.UNIFORM_FILTERS or mode==Mode.UNIFORM_LAYER:
            show_channel(l_to_plot_idx, 0, rec.layers_layout[l_to_plot_idx], l_to_plot[0], rec.patch_size)
        else:
            show_layer(l_to_plot_idx, rec.layers_layout[l_to_plot_idx], l_to_plot)
            show_channel(l_to_plot_idx, 0, rec.layers_layout[l_to_plot_idx], l_to_plot[0], rec.patch_size)
            show_channel(l_to_plot_idx, round(rec.layers_layout[l_to_plot_idx][0]/2), rec.layers_layout[l_to_plot_idx], 
                         l_to_plot[round(rec.layers_layout[l_to_plot_idx][0]/2)], rec.patch_size)
            show_channel(l_to_plot_idx, rec.layers_layout[l_to_plot_idx][0]-1, rec.layers_layout[l_to_plot_idx], 
                         l_to_plot[rec.layers_layout[l_to_plot_idx][0]-1], rec.patch_size)
    return rec
    
def plot_ops_saved_vs_ones(net_name, dataset_name, ps, ones_possibilities, gran_thresh, acc_loss, init_acc, mode=None):
    bs_line_rec = get_baseline_rec(net_name, dataset_name, ps, init_acc)
    plt.figure()
    if bs_line_rec is not None:
        plt.plot(ones_possibilities, [round(bs_line_rec.ops_saved/bs_line_rec.total_ops, 3)]*len(ones_possibilities),'o--', label='baseline')
    
    modes = get_modes(mode)
    for mode in modes:
        ops_saved = [None]*len(ones_possibilities)
        for idx, ones in enumerate(ones_possibilities):
            rec_finder = RecordFinder(net_name, dataset_name, ps, (ones,ones+1), gran_thresh, acc_loss, init_acc)
            fn = rec_finder.find_rec_filename(mode, RecordType.FINAL_RESULT_REC)
            rec = load_from_file(fn,'')
            ops_saved[idx] = round(rec.ops_saved/rec.total_ops, 3)
        plt.plot(ones_possibilities, ops_saved,'o--', label=gran_dict[mode])
    plt.xlabel('number of ones') 
    plt.ylabel('operations saved [%]') 
    plt.title(f'Operations Saved vs Number of Ones \n'
              f'{net_name}, {dataset_name}, INITIAL ACC:{init_acc} \n'
              f'PATCH SIZE:{ps}, MAX ACC LOSS:{acc_loss}, GRANULARITY:{gran_thresh}')
    plt.legend()
    plt.savefig(f'{cfg.RESULTS_DIR}ops_saved_vs_number_of_ones_{net_name}_{dataset_name}'+
                f'acc{init_acc}_ps{ps}_ma{acc_loss}_mg{gran_thresh}.pdf')
    
def plot_ops_saved_vs_max_acc_loss(net_name, dataset_name, ps, ones_range, gran_thresh, acc_loss_opts, init_acc, mode=None):
    bs_line_rec = get_baseline_rec(net_name, dataset_name, ps, init_acc)
    plt.figure()
    if bs_line_rec is not None:
        plt.plot(acc_loss_opts, [round(bs_line_rec.ops_saved/bs_line_rec.total_ops, 3)]*len(acc_loss_opts),'o--', label='baseline')

    rec_finder = RecordFinder(net_name, dataset_name, ps, ones_range, gran_thresh, '*', init_acc)
    modes = get_modes(mode)   
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

def show_channel(layer, channel, dims, image, ps, filename=None):
    plt.figure()
    plt.imshow(image)
    plt.title(f'Layer:{layer}, Channel:{channel}, dims:{dims}')
    ax = plt.gca()
    # Minor ticks
    ax.set_xticks(np.arange(-.5, image.shape[0]-1, ps), minor=True);
    ax.set_yticks(np.arange(-.5, image.shape[1]-1, ps), minor=True);
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    plt.tick_params(axis='both', which='major', bottom=False, top=False,
                    left=False, right=False, labelbottom=False, labelleft=False)
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

def get_modes(mode):
    if mode is None:
        modes = [m for m in Mode]
    elif type(mode)==list:
        modes = mode
    else:
        modes = [mode]
    return modes

def get_baseline_rec(net_name, dataset_name, ps, init_acc):
    rec_finder = RecordFinder(net_name, dataset_name, ps, ('*','*'), '*', '*', init_acc)
    bs_line_fn = rec_finder.find_rec_filename(None, RecordType.BASELINE_REC)
    if bs_line_fn is None:
        optim = Optimizer(ps, (None,None), None, None)
        optim.base_line_result()
        bs_line_fn = rec_finder.find_rec_filename(None, RecordType.BASELINE_REC)
    if bs_line_fn is None:
        print(f' !!! Was not able to get baseline result for initial accuracy of {init_acc} !!!')
        print(f' !!! Adjust TEST_SET_SIZE in Config.py !!!')
        return bs_line_fn
    return load_from_file(bs_line_fn, '')
