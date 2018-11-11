# ----------------------------------------------------------------------------------------------------------------------
#                                                     Imports
# ----------------------------------------------------------------------------------------------------------------------
# Torch Libraries:
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

# Models: (There are many more)
from models.resnet import ResNet18
from models.googlenet import GoogleNet
from models.densenet import DenseNet121
from models.vgg import VGG #VGG('VGG19')

# Utils:
from run_config import RunConfig,RunDefaults
from tqdm import tqdm
import os
from utils.fs import enable_dir
from utils.data_import import CIFAR10
# from pickle import load, dump
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------


from keras.utils import generic_utils
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def main():

    rc = RunConfig(ResNet18,CIFAR10)


    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)






class NeuralNet:
    def __init__(self):
        self.rc = RunConfig()
        self.defs = RunDefaults()
        self.best_acc = self.rc.record_acc()

        net = nn_creator()
        data =
        net = net.to(device)

        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True



        # INIT TEST LOADER / NET

    def train(self):
        for epoch in range(start_epoch, start_epoch + 200):
            # TRAIN
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    def test(self):

        self.net.eval() #Notify layers that we are testing

        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.rc.device()), targets.to(self.rc.device())
                outputs = self.net(inputs)
                loss = self.rc.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': self.epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')
            best_acc = acc

    def checkpoint(self):
        state = {
            'net': self.net.state_dict(),
            'acc': self.acc,
            'epoch': self.epoch,
        }
        enable_dir(self.defs.checkpoint_path())
        torch.save(state,os.path.join(self.defs.checkpoint_path(),'TMP.t7'))


if __name__ == '__main__': main()
