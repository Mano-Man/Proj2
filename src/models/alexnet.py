import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from .SpatialNet import Spatial, SpatialNet,sequential_spatial_layer_extract
import torch.nn.functional as tf

class AlexNetS(SpatialNet):

    def __init__(self,device, output_channels, input_channels,data_shape):
        super().__init__(device)

        if data_shape[1] <= 64:
            using_small = True
            params = [5,2,2,2]
        else:
            using_small = False
            params = [2,3,3,3]

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=params[0]), # padding=2
            nn.ReLU(inplace=True),
            Spatial(64),
            nn.MaxPool2d(kernel_size=params[1], stride=2), #Kernel size = 3
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            Spatial(192),
            nn.MaxPool2d(kernel_size=params[2], stride=2), #kernel size = 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Spatial(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Spatial(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Spatial(256),
            nn.MaxPool2d(kernel_size=params[3], stride=2),#kernel size = 3

        )
        # This is a dirty hack to fit the dense layer to any needed size.
        self.classifier = lambda x: x
        linear_size = self.output_size(x_shape=data_shape,cuda_allowed=False)

        if using_small:
            self.classifier = nn.Linear(linear_size, output_channels)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(linear_size, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, output_channels),
            )

        self.spatial_layers = sequential_spatial_layer_extract(self.features)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        #print(x.size())
        x = self.classifier(x)
        return x

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    # TODO - Implement pretrained support for Spatial
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'))
    return model