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
# from models.googlenet import GoogleNet
# from models.densenet import DenseNet121
# from models.vgg import VGG  # VGG('VGG19')

# Utils:
# from run_config import RunConfig, RunDefaults
# from tqdm import tqdm
# from pickle import load, dump

import os
from util.data_import import CIFAR10_Train, CIFAR10_Test
from util.gen import Progbar, banner
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

VERBOSITY = 1  # 0 for per epoch output, 1 for per-batch output

RESUME_CHECKPOINT = True
N_EPOCHS = 1
LEARN_RATE = 0.01
BATCH_SIZE = 128
TRAIN_SET_SIZE = 10000  # Max 50000


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def main():
    nn = NeuralNet()
    nn.train(N_EPOCHS)
    test_gen = CIFAR10_Test(batch_size=BATCH_SIZE)
    test_loss, test_acc, count = nn.test(test_gen)
    print(f'==> Final testing results: test acc: {test_acc:.3f} with {count}, test loss: {test_loss:.3f}')


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class NeuralNet:
    def __init__(self):

        # Decide on device:
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            print('Warning: Found no valid GPU device - Running on CPU')

        # Build Model:
        print('==> Building  model..')
        self.net = ResNet18()
        self.net = self.net.to(self.device)
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True

        # Build SGD Algorithm:
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=LEARN_RATE, momentum=0.9, weight_decay=5e-4)

        if RESUME_CHECKPOINT:
            print('==> Resuming from checkpoint')
            assert os.path.isdir('./data/checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load('./data/checkpoint/ckpt.t7')
            self.net.load_state_dict(checkpoint['net'])
            self.best_val_acc = checkpoint['acc']
            print(f'==> Loaded model with val-acc of {self.best_val_acc}')
            self.start_epoch = checkpoint['epoch']
        else:
            self.best_val_acc = 0
            self.start_epoch = 0

        # Init Data:
        self.train_gen, self.val_gen, self.classes = CIFAR10_Train(batch_size=BATCH_SIZE, dataset_size=TRAIN_SET_SIZE)

    def train(self, n_epochs):
        if (VERBOSITY > 0):
            p = Progbar(n_epochs)
        for epoch in range(self.start_epoch, self.start_epoch + n_epochs):
            banner(f'Epoch: {epoch}')
            train_loss, train_acc, train_count = self._train_step()
            val_loss, val_acc, val_count = self.test(self.val_gen)
            self._checkpoint(val_acc, epoch + 1)
            if (VERBOSITY > 0):
                p.add(1,
                      values=[("t_loss", train_loss), ("t_acc", train_acc), ("v_loss", val_loss), ("v_acc", val_acc)])
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

    def _checkpoint(self, acc, epoch):
        if acc > self.best_val_acc:
            print(f'Beat val_acc record of {self.best_val_acc} with {acc} - Saving checkpoint')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('./data/checkpoint'):
                os.mkdir('./data/checkpoint')
            torch.save(state, './data/checkpoint/ckpt.t7')
            self.best_acc = acc

    def _train_step(self):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0

        epoch_progress = Progbar(len(self.train_gen))
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

            epoch_progress.add(1, values=[("t_loss", train_loss / (batch_idx + 1)), ("t_acc", 100. * correct / total)])

        total_acc = 100. * correct / total
        count = f'{correct}/{total}'
        return train_loss, total_acc, count


if __name__ == '__main__':
    main()
