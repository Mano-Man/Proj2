import os
import urllib

# Torch Libraries:
import torch
import torchvision
import torchvision.transforms as transforms

# ----------------------------------------------------------------------------------------------------------------------
#                                               	 Torch
# ----------------------------------------------------------------------------------------------------------------------
def CIFAR10(save_path='./data'):
    """
    :param save_path: Where to save the data location for future usage
    :return: A train + test generator, along with the 10 class names in a tuple structure
    """
    print('==> Preparing CIFAR10 data..')
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

    trainset = torchvision.datasets.CIFAR10(root=save_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=save_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader,testloader,classes

# ----------------------------------------------------------------------------------------------------------------------
#                                               	General
# ----------------------------------------------------------------------------------------------------------------------
def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    # Usage : 
    #   url = 'http://mattmahoney.net/dc/'
    #   filename = maybe_download('text8.zip', url, 31344016)
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


