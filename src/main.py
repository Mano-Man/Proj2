# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
# Torch Libraries:
import torch
import torch.nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Models: (There are many more)
from models.resnet import ResNet18
from models.resnet_spatial import ResNet18Spatial, ResNet34Spatial

# Utils:
import re
import os
import glob
from tqdm import tqdm
from util.data_import import CIFAR10_Train, CIFAR10_Test
from util.gen import Progbar, banner, dict_sym_diff

import Record as rc
import maskfactory as mf

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Global Config
# ----------------------------------------------------------------------------------------------------------------------
# Global Adjustments:
NETS = [ResNet18, ResNet18Spatial, ResNet34Spatial]
NET = NETS[1]
BATCH_SIZE = 128

# Verbosity Adjustments:
VERBOSITY = 1  # 0 for per epoch output, 1 for per-batch output
DO_DOWNLOAD = False

# Checkpoint Adjustments
RESUME_CHECKPOINT = True
RESUME_METHOD = 'ValAcc'  # 'ValAcc' 'Time'
CHECKPOINT_DIR = './data/checkpoint/'  # '/content/drive/My Drive/Colab Notebooks/data/checkpoint/'
RESULTS_DIR = './data/results/'  # '/content/drive/My Drive/Colab Notebooks/data/results/'
# ----------------------------------------------------------------------------------------------------------------------
#                                               Spatial Functionality
# ----------------------------------------------------------------------------------------------------------------------
# Spatial Config
RECORDS_FILENAME = ''
GEN_PATTERNS = True
ONES_RANGE = (1, 2)  # Exclusive range

MODE = rc.uniform_layer
LAYER_LAYOUT = rc.Resnet18_layers_layout
PS = 2
GRAN_THRESH = 32 * 32

# Complexity Config
SAVE_INTERVAL = 100
RESUME_MASK_GEN = False

# Dummy workloads
SP_MOCK = [(0, 0, 0)] * len(LAYER_LAYOUT)
SP_ZEROS = [(1, PS, torch.zeros([64, 32, 32])),
            (1, PS, torch.zeros([64, 32, 32])),
            (1, PS, torch.zeros([128, 16, 16])),
            (1, PS, torch.zeros([256, 8, 8])),
            (1, PS, torch.zeros([512, 4, 4]))]

CHOSEN_SP = SP_MOCK
# ----------------------------------------------------------------------------------------------------------------------
#                                                Train Functionality
# ----------------------------------------------------------------------------------------------------------------------
# Train Adjustments - TODO - Write a learning step decrease functionality
N_EPOCHS = 30
TRAIN_SET_SIZE = 50000  # Max 50000 for CIFAR10
LEARN_RATE = 0.1
DONT_SAVE_REDUNDANT = True  # Don't checkpoint if val_acc achieved is lower than what is in the cp directory


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def main():
    nn = NeuralNet(CHOSEN_SP)
    test_gen = CIFAR10_Test(batch_size=BATCH_SIZE, download=DO_DOWNLOAD)
    _, test_acc, _ = nn.test(test_gen)
    print(f'==> Asserted test-acc of: {test_acc}\n')

    if RESUME_MASK_GEN:
        rcs = rc.load_from_file(RECORDS_FILENAME, path=RESULTS_DIR)
        st_point = rcs.find_resume_point()
    else:
        rcs = rc.Record(LAYER_LAYOUT, GRAN_THRESH, GEN_PATTERNS, MODE, test_acc, PS, ONES_RANGE)
        st_point = [0] * 4
    rcs.filename = 'ps' + str(PS) + '_ones' + str(ONES_RANGE) + '_' + rcs.filename

    save_counter = 0
    for layer, channel, patch, pattern_idx, mask in tqdm(mf.gen_masks_with_resume \
                                                                     (PS, rcs.all_patterns, rcs.mode, rcs.gran_thresh,
                                                                      LAYER_LAYOUT, resume_params=st_point)):
        sp_list = SP_MOCK
        sp_list[layer] = (1, PS, torch.from_numpy(mask))
        nn.update_spatial(sp_list)
        _, test_acc, _ = nn.test(test_gen)
        ops_saved, ops_total = nn.net.num_ops()
        rcs.addRecord(ops_saved.item(), ops_total, test_acc, layer, channel, patch, pattern_idx)

        save_counter += 1
        if save_counter > SAVE_INTERVAL:
            rc.save_to_file(rcs, True, RESULTS_DIR)
            save_counter = 0

    rc.save_to_file(rcs, True, RESULTS_DIR)
    rcs.save_to_csv(RESULTS_DIR)
    print('==> Result saved to ' + os.path.join(RESULTS_DIR, rcs.filename))


def training_main():
    nn = NeuralNet(CHOSEN_SP)
    nn.train(N_EPOCHS)
    test_gen = CIFAR10_Test(batch_size=BATCH_SIZE, download=DO_DOWNLOAD)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class NeuralNet:
    def __init__(self, sp_list):

        # Decide on device:
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
            if torch.cuda.device_count() > 1:
                raise AssertionError
                # This line enables multiple GPUs, but changes the layer names a bit
                # self.net = torch.nn.DataParallel(self.net)  # Useful if you have multiple GPUs - does not hurt otherwise
        else:
            self.device = torch.device('cpu')
            # torch.set_num_threads(4) # Presuming 4 cores
            print('WARNING: Found no valid GPU device - Running on CPU')

        # Build Model:
        print(f'==> Building model {NET.__name__}')
        self.net = NET(sp_list)

        if RESUME_CHECKPOINT:
            print(f'==> Resuming from checkpoint via sorting method: {RESUME_METHOD}')
            assert os.path.isdir(CHECKPOINT_DIR), 'Error: no checkpoint directory found!'
            if RESUME_METHOD == 'Time':
                ck_file = self._find_latest_checkpoint()
            elif RESUME_METHOD == 'ValAcc':
                ck_file = self._find_top_val_acc_checkpoint()
            else:
                raise AssertionError

            if ck_file is None:
                print(f'Found no valid checkpoints for {NET.__name__}')
                self.best_val_acc = 0
                self.start_epoch = 0
            else:
                checkpoint = torch.load(ck_file, map_location=self.device)
                self._load_checkpoint(checkpoint['net'])
                self.best_val_acc = checkpoint['acc']
                self.start_epoch = checkpoint['epoch']
                print(f'==> Loaded model with val-acc of {self.best_val_acc}')

        else:
            self.best_val_acc = 0
            self.start_epoch = 0

        self.net = self.net.to(self.device)

        # Build SGD Algorithm:
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=LEARN_RATE,
                                   momentum=0.9, weight_decay=5e-4)

    def update_spatial(self, sp_list):
        old_state = self.net.state_dict()
        self.net = NET(sp_list)  # TODO: This update can be done without a full reconstruction
        self.net.load_state_dict(old_state,strict=False)
        self.net = self.net.to(self.device)
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=LEARN_RATE,
                                   momentum=0.9, weight_decay=5e-4)
        return self

    def train(self, n_epochs):

        # Bring in Data
        self.train_gen, self.val_gen, self.classes = CIFAR10_Train(batch_size=BATCH_SIZE, dataset_size=TRAIN_SET_SIZE,
                                                                   download=DO_DOWNLOAD)
        p = Progbar(n_epochs)
        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            if VERBOSITY > 0:
                banner(f'Epoch: {epoch}')
            train_loss, train_acc, train_count = self._train_step()
            val_loss, val_acc, val_count = self.test(self.val_gen)
            p.add(1, values=[("t_loss", train_loss), ("t_acc", train_acc), ("v_loss", val_loss), ("v_acc", val_acc)])
            self._checkpoint(val_acc, epoch + 1)
        banner('Training Phase - End')

    def test(self, data_gen):

        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_gen):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # val_loss = f'{test_loss/(batch_idx+1):.3f}.'
                # acc = f'{100.*correct/total:.3f}.'
                # count = f'{correct}/{total}'

        test_acc = 100. * correct / total
        count = f'{correct}/{total}'

        return test_loss, test_acc, count

    def _checkpoint(self, val_acc, epoch):

        # Decide on whether to checkpoint or not:
        save_it = val_acc > self.best_val_acc
        if save_it and DONT_SAVE_REDUNDANT:
            checkpoints = [os.path.basename(f) for f in glob.glob(f'{CHECKPOINT_DIR}{NET.__name__}_*_ckpt.t7')]
            if checkpoints:
                best_cp_val_acc = max([float(f.replace(NET.__name__, '').split('_')[1]) for f in checkpoints])
                if best_cp_val_acc >= val_acc:
                    save_it = False
                    print(f'Resuming without save - Found valid checkpoint with higher val_acc: {best_cp_val_acc}')
        # Do checkpoint
        if save_it:
            print(f'\nBeat val_acc record of {self.best_val_acc} with {val_acc} - Saving checkpoint')
            state = {
                'net': self.net.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
            if not os.path.isdir(CHECKPOINT_DIR):
                os.mkdir(CHECKPOINT_DIR)
            torch.save(state, f'{CHECKPOINT_DIR}{NET.__name__}_{val_acc}_ckpt.t7')
            self.best_val_acc = val_acc

    def _train_step(self):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        if VERBOSITY > 0:
            prog_batch = Progbar(len(self.train_gen))
        for batch_idx, (inputs, targets) in enumerate(self.train_gen):
            # Training step
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Collect results:
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if VERBOSITY > 0:
                prog_batch.add(1, values=[("t_loss", train_loss / (batch_idx + 1)), ("t_acc", 100. * correct / total)])

        total_acc = 100. * correct / total
        count = f'{correct}/{total}'
        return train_loss, total_acc, count

    def _load_checkpoint(self, loaded_dict, optional_fill=['.*\.num_batches_tracked'], total_ignore=['.*pred\.']):

        # Make a regex that matches if any of our regexes match.
        opt_fill = "(" + ")|(".join(optional_fill) + ")"
        tot_ignore = "(" + ")|(".join(total_ignore) + ")"

        curr_dict = self.net.state_dict()
        filtered_dict = {}

        for k, v in loaded_dict.items():
            if not re.match(tot_ignore, k):  # If in ignore list, ignore
                if k in curr_dict:
                    filtered_dict[k] = v
                else:  # Check if it is possible to ignore it being gone
                    if not re.match(opt_fill, k):
                        assert False, f'Fatal: found unknown entry {k} in loaded checkpoint'

        assert filtered_dict, 'State dictionary is empty'
        # Also check for missing entries in loaded checkpoint
        for k, v in curr_dict.items():
            if k not in loaded_dict and not (re.match(opt_fill, k) or re.match(tot_ignore, k)):
                assert False, f'Fatal: missing entry {k} from checkpoint'

        # Overwrite entries in the existing state dict
        curr_dict.update(filtered_dict)
        self.net.load_state_dict(curr_dict)

    @staticmethod
    def _find_top_val_acc_checkpoint():

        checkpoints = [os.path.basename(f) for f in glob.glob(f'{CHECKPOINT_DIR}{NET.__name__}_*_ckpt.t7')]
        if not checkpoints:
            return None
        else:
            checkpoints.sort(key=lambda x: x.replace(NET.__name__, '').split('_')[1])
            # print(checkpoints)
            return CHECKPOINT_DIR + checkpoints[-1]

    @staticmethod
    def _find_latest_checkpoint():

        checkpoints = glob.glob(f'{CHECKPOINT_DIR}{NET.__name__}_*_ckpt.t7')
        if not checkpoints:
            return None
        else:
            checkpoints.sort(key=os.path.getmtime)
            # print(checkpoints)
            return checkpoints[-1]


if __name__ == '__main__':
    main()
