# Models: (There are many more)
from models.resnet import ResNet18, ResNet18Spatial, ResNet34Spatial
from util.datasets import CIFAR10_shape, CIFAR10_train, CIFAR10_test, ImageNet_shape, ImageNet_train, ImageNet_test
import os

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Global Config
# ----------------------------------------------------------------------------------------------------------------------
# Global Adjustments:
NETS = [ResNet18, ResNet18Spatial, ResNet34Spatial]

DATA_SETS =	{
  "ImageNet": (ImageNet_shape, ImageNet_train, ImageNet_test),
  "CIFAR10": (CIFAR10_shape, CIFAR10_train, CIFAR10_test)
}

BATCH_SIZE = 128
TEST_SET_SIZE = 1000 #BATCH_SIZE * 8 # This is 1024 - Max for CIFAR10 is 10000  - Better to align it to Batch Size for speed!
 
NET = NETS[1]  # The chosen network
DATA_NAME = "CIFAR10" # The chosen data

# Complexity Config
SAVE_INTERVAL = 100
CHANNELQ_UPDATE_RATIO = 1
PATCHQ_UPDATE_RATIO = 1


# ----------------------------------------------------------------------------------------------------------------------
#                                                   Data Import
# ----------------------------------------------------------------------------------------------------------------------

CHECKPOINT_DIR = './data/checkpoint/'
RESULTS_DIR = './data/results/'
DO_DOWNLOAD = False
if not os.path.isdir(CHECKPOINT_DIR):
    CHECKPOINT_DIR = '/content/drive/My Drive/Colab Notebooks/data/checkpoint/'
    RESULTS_DIR = '/content/drive/My Drive/Colab Notebooks/data/results/'
    DO_DOWNLOAD = True
# ----------------------------------------------------------------------------------------------------------------------
#                                                Train Specific Functionality
# ----------------------------------------------------------------------------------------------------------------------
# Verbosity Adjustments:
VERBOSITY = 1  # 0 for per epoch output, 1 for per-batch output

# Checkpoint Adjustments
RESUME_CHECKPOINT = True
RESUME_METHOD = 'ValAcc'  # 'ValAcc' 'Time'

# Train Adjustments - Notice that there is currently no learning step decrease functionality
N_EPOCHS = 30
TRAIN_SET_SIZE = 5000  # Max 50000 for CIFAR10 , Max ???? for ImageNet
LEARN_RATE = 0.1
DONT_SAVE_REDUNDANT = True  # Don't checkpoint if val_acc achieved is lower than what is in the cp directory
# ----------------------------------------------------------------------------------------------------------------------
#                                                    For ease of access
# ----------------------------------------------------------------------------------------------------------------------
DATA_SHAPE = DATA_SETS[DATA_NAME][0]
TRAIN_GEN = DATA_SETS[DATA_NAME][1]
TEST_GEN = DATA_SETS[DATA_NAME][2]
