# Models: (There are many more)
from models.resnet import ResNet18, ResNet18Spatial, ResNet34Spatial
from models.alexnet import AlexNetS
from util.datasets import Datasets
import os

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Data Import
# ----------------------------------------------------------------------------------------------------------------------

if os.path.isdir('/content/drive/My Drive/Colab Notebooks/'):
    basedir = '/content/drive/My Drive/Colab Notebooks/data'
else:
    basedir, _ = os.path.split(os.path.abspath(__file__))
    basedir = os.path.join(basedir, 'data')

CHECKPOINT_DIR = os.path.join(basedir, 'checkpoint')
RESULTS_DIR = os.path.join(basedir, 'results')
DATASET_DIR = os.path.join(basedir, 'datasets')

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Global Config
# ----------------------------------------------------------------------------------------------------------------------
# Global Adjustments:
NETS = [ResNet18, ResNet18Spatial, ResNet34Spatial, AlexNetS]
NET = NETS[2]  # The chosen network

# print(Datasets.which()) # ('MNIST', 'CIFAR10', 'ImageNet','STL10)
DATA = Datasets.get('CIFAR10')

BATCH_SIZE = 128
TEST_SET_SIZE = BATCH_SIZE * 8  # Better to align it to Batch Size for speed!

# Complexity Config
SAVE_INTERVAL = 100
# ----------------------------------------------------------------------------------------------------------------------
#                                                Train Specific Functionality
# ----------------------------------------------------------------------------------------------------------------------
VERBOSITY = 1  # 0 for per epoch output, 1 for per-batch output
RESUME_METHOD = 'ValAcc'  # 'ValAcc' 'Time'
DONT_SAVE_REDUNDANT = True  # Don't checkpoint if val_acc achieved is lower than what is in the cp directory
