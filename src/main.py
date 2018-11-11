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
# from models.googlenet import GoogleNet
# from models.densenet import DenseNet121
# from models.vgg import VGG  # VGG('VGG19')

# Utils:
# from run_config import RunConfig, RunDefaults
from tqdm import tqdm
import os
from util.fs import enable_dir
from util.data_import import CIFAR10
from util.gen import Progbar

# from pickle import load, dump
# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    trainloader, testloader, classes = CIFAR10()

    # Model
    print('==> Building model..')
    net = ResNet18()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if 0:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    NUM_EPOCHS = 200
    progbar = Progbar(NUM_EPOCHS)
    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        train_loss, train_acc, train_count = train(epoch, net, optimizer, device, trainloader, criterion)
        val_loss, val_acc, val_count = test(epoch, net, device, testloader, criterion, best_acc)
        progbar.add(1, values=[("t_loss", train_loss), ("t_acc", train_acc),
                               ("v_loss", val_loss), ("v_acc", val_acc)])


# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------
class NeuralNet:
    def __init__(self):
        #self.rc = RunConfig()
        #self.defs = RunDefaults()
        #self.best_acc = self.rc.record_acc()

        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

    def train(self):


    def test(self):


        net = nn_creator()
        data =
        net = net.to(device)

        if self.device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True


# Training
def train(epoch, net, optimizer, device, trainloader, criterion):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    progbar = Progbar(len(trainloader))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
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

        train_loss = float(f'{train_loss/(batch_idx+1):.3f}')
        acc = float(f'{100.*correct/total:.3f}')
        progbar.add(1, values=[("t_loss", train_loss), ("t_acc", acc)])

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    acc = 100. * correct / total
    count = f'{correct}/{total}'
    return train_loss, acc, count


def test(epoch, net, device, testloader, criterion, best_acc):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # val_loss = f'{test_loss/(batch_idx+1):.3f}.'
            # acc = f'{100.*correct/total:.3f}.'
            # count = f'{correct}/{total}'

    # Save checkpoint.
    acc = 100. * correct / total
    count = f'{correct}/{total}'
    if acc > best_acc:
        print(f'Beat val_acc record of {best_acc} with {acc} - Saving checkpoint')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
    return test_loss, acc, count


if __name__ == '__main__': main()
