# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:52:19 2018

@author: Inna
"""
import torch
import torch.nn as nn
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

from models.resnet_spatial import ResNet18Spatial
from models.resnet_spatial import expand_mat
from models.resnet_spatial import expand_mat_gpu


if __name__ == "__main__":
    device = 'cpu'
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
#    for i_batch, sample_batched in enumerate(trainloader):
#        if i_batch > 0:
#            break
#        print(sample_batched[0].shape)
#        print(sample_batched[1].shape)
        
    # Model
    print('==> Building model..')
    patch_size = 2
    sp0 = (patch_size, torch.ones(64, 32, 32))
    sp1 = (patch_size, torch.ones(64, 32, 32))
    sp2 = (patch_size, torch.ones(128, 16, 16))
    sp3 = (patch_size, torch.ones(256, 8, 8))
    sp4 = (patch_size, torch.ones(512, 4, 4))
    sp_list = [sp0, sp1, sp2, sp3, sp4]

    net = ResNet18Spatial(sp_list, pretrained=True)

    print(net)
    summary(net, (3, 32, 32))
    print('==> model built..')
    
#    a = torch.Tensor([[0, 1, 0], [1, 0, 1],[1,1,1]])
#    ex = a.unsqueeze(0).repeat(3, 1, 1)
#    ex = a.unsqueeze(0).repeat(2, 3, 1, 1)
#    a_exp = expand_mat(ex, 2, 4)
#    a_exp_gpu = expand_mat_gpu(ex, 2, 4)
    
