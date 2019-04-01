# Models: (There are many more)
from enum import Enum
from models.resnet import ResNet18, ResNet18Spatial,ResNet18SpatialUniBlock,ResNet18SpatialUniCluster, ResNet34Spatial
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
NETS = [ResNet18Spatial,ResNet18SpatialUniBlock,ResNet18SpatialUniCluster, ResNet34Spatial, AlexNetS]
NET = NETS[0]  # The chosen network

#print(Datasets.which()) #('MNIST', 'CIFAR10', 'ImageNet', 'TinyImageNet', 'STL10', 'FashionMNIST')
DATA = Datasets.get('CIFAR10',DATASET_DIR)

BATCH_SIZE = 128
TEST_SET_SIZE = 1000 #BATCH_SIZE * 8  # Better to align it to Batch Size for speed!

# Complexity Config
SAVE_INTERVAL = 100
CHANNELQ_UPDATE_RATIO = 1
PATCHQ_UPDATE_RATIO = 1

# for future debuging...
TWO_STAGE = True

class LQ_modes(Enum):
    DEFAULT = 10
    PRODUCT = 12
    CLEAN_DECREASING_ACC = 17
class CQ_modes(Enum):
    DEFAULT = 1
class PQ_modes(Enum):
    DEFAULT = 1
    
LQ_OPTION = LQ_modes.DEFAULT
CQ_OPTION = CQ_modes.DEFAULT
PQ_OPTION = PQ_modes.DEFAULT

# ------------------------------, ----------------------------------------------------------------------------------------
#                                                Train Specific Functionality
# ----------------------------------------------------------------------------------------------------------------------
N_EPOCHS_TO_WAIT_BEFORE_LR_DECAY = 3
SGD_METHOD = 'Nesterov' # Can also use: 'Adam' 'Nesterov'
VERBOSITY = 1  # 0 for per epoch output, 1 for per-batch output
RESUME_METHOD = 'ValAcc'  # 'ValAcc' 'Time'
DONT_SAVE_REDUNDANT = True  # Don't checkpoint if val_acc achieved is lower than what is in the cp directory
