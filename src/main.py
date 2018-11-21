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
from models.resnet_spatial import ResNet18Spatial
# from models.googlenet import GoogleNet
# from models.densenet import DenseNet121
# from models.vgg import VGG  # VGG('VGG19')

# Utils:
# from run_config import RunConfig, RunDefaults
# from tqdm import tqdm
# from pickle import load, dump

import os
import glob
from util.data_import import CIFAR10_Train, CIFAR10_Test
from util.gen import Progbar, banner

import Record as rc
import maskfactory as mf

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
# Global Adjustments:
NET = ResNet18Spatial   # ResNet18 ResNet18Spatial, Data is currently hard-coded

# Verbosity Adjustments:
VERBOSITY = 1  # 0 for per epoch output, 1 for per-batch output
DO_DOWNLOAD = True

# Train Adjustments
N_EPOCHS = 30
LEARN_RATE = 0.1
BATCH_SIZE = 128
# TODO - Write a learning step decrease functionality
TRAIN_SET_SIZE = 50000  # Max 50000

# Checkpoint Adjustments
RESUME_CHECKPOINT = True
RESUME_METHOD = 'ValAcc'  # 'ValAcc' 'Time'
CHECKPOINT_DIR = '/content/drive/My Drive/Colab Notebooks/data/checkpoint/'
RESULTS_DIR = '/content/drive/My Drive/Colab Notebooks/data/results/'
#CHECKPOINT_DIR = './data/checkpoint/'
#RESULTS_DIR = './data/results/'
DONT_SAVE_REDUNDANT = True  # Don't checkpoint if val_acc achieved is lower than what is in the cp directory


#SP_LIST = [(1, PATCH_SIZE, torch.ones([64, 32, 32])),
#           (1, PATCH_SIZE, torch.ones([64, 32, 32])),
#           (1, PATCH_SIZE, torch.ones([128, 16, 16])),
#           (1, PATCH_SIZE, torch.ones([256, 8, 8])),
#           (1, PATCH_SIZE, torch.ones([512, 4, 4]))]



PATCH_SIZE = 2
MAX_GRA = 32*32
GENERATE_PATTERNS = True
MIN_ONES = 1
MAX_ONES = MIN_ONES + 1
LAYER_LAYOUT = rc.Resnet18_layers_layout
MODE = rc.uniform_layer
SP_LIST_DISABLE = [(0, PATCH_SIZE, torch.zeros(0))]*len(LAYER_LAYOUT)
SAVE_INTERVAL = 100
RESUME_MASK_GEN = False
RECORDS_FILENAME = ''

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def training_main():
    nn = NeuralNet(SP_LIST_DISABLE)
    nn.train(N_EPOCHS)
    test_gen = CIFAR10_Test(batch_size=BATCH_SIZE, download=DO_DOWNLOAD)  # Check the case when we don't download!
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')

def main():
    test_gen = CIFAR10_Test(batch_size=BATCH_SIZE, download=DO_DOWNLOAD)
    nn = NeuralNet(SP_LIST_DISABLE)
    test_loss, initial_acc, count = nn.test(test_gen)
    print(f'=====>  loaded model results: initial acc {initial_acc:.3f} with {count}, initial loss: {test_loss:.3f}')
    if RESUME_MASK_GEN:
        records = rc.load_from_file(RECORDS_FILENAME, path=RESULTS_DIR)
        st_point = records.find_resume_point()
    else:
        records = rc.Record(LAYER_LAYOUT,MAX_GRA,GENERATE_PATTERNS, MODE,initial_acc, \
                        PATCH_SIZE,MIN_ONES,MAX_ONES)
        st_point = [0]*4
    records.filename = 'ps'+str(PATCH_SIZE)+'_ones'+ str(MIN_ONES)+'_'+records.filename
    print('=====> result will be saved to ' + os.path.join(RESULTS_DIR, records.filename))
    
    #test ----------------------------------
#    sp_list = [(1, PATCH_SIZE, torch.zeros([64, 32, 32])),
#           (1, PATCH_SIZE, torch.zeros([64, 32, 32])),
#           (1, PATCH_SIZE, torch.zeros([128, 16, 16])),
#           (1, PATCH_SIZE, torch.zeros([256, 8, 8])),
#           (1, PATCH_SIZE, torch.zeros([512, 4, 4]))]
#    nn = NeuralNet(sp_list)
#    test_loss, test_acc, count = nn.test(test_gen)
#    ops_saved, ops_total = nn.net.no_of_operations()
#    records.addRecord(ops_saved.item(), ops_total, test_acc, 0, 0, 0, 0)
    
    save_counter = 0
    for layer, channel, patch, pattern_idx, mask in mf.gen_masks_with_resume(PATCH_SIZE,  \
                                                                             records.all_patterns, \
                                                                             records.mode, \
                                                                             records.max_gra, \
                                                                             LAYER_LAYOUT, \
                                                                             resume_params=st_point):
        sp_list = [(0, PATCH_SIZE, torch.zeros(0))]*len(LAYER_LAYOUT)
        sp_list[layer] = (1, PATCH_SIZE, torch.from_numpy(mask))
        nn = NeuralNet(sp_list)
        test_loss, test_acc, count = nn.test(test_gen)
        ops_saved, ops_total = nn.net.no_of_operations()
        records.addRecord(ops_saved.item(), ops_total, test_acc, layer, channel, patch, pattern_idx)
        
        save_counter += 1
        if save_counter > SAVE_INTERVAL:
            rc.save_to_file(records, True, RESULTS_DIR)
            save_counter = 0
            
    rc.save_to_file(records, True, RESULTS_DIR)
    records.save_to_csv(RESULTS_DIR)
    print('=====> result saved to ' + os.path.join(RESULTS_DIR, records.filename))    
    
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
            #torch.set_num_threads(4) # Presuming 4 cores
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
                self.net.load_state_dict(checkpoint['net'])
                self.best_val_acc = checkpoint['acc']
                self.start_epoch = checkpoint['epoch']
                print(f'==> Loaded model with val-acc of {self.best_val_acc}')

        else:
            self.best_val_acc = 0
            self.start_epoch = 0


        self.net = self.net.to(self.device)

        # Build SGD Algorithm:
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=LEARN_RATE, momentum=0.9, weight_decay=5e-4)

        # Bring in Data
        self.train_gen, self.val_gen, self.classes = CIFAR10_Train(batch_size=BATCH_SIZE, dataset_size=TRAIN_SET_SIZE,
                                                                   download=DO_DOWNLOAD)

    def train(self, n_epochs):

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
