import torch
# Models: (There are many more)
from models.resnet import ResNet18
from models.resnet_spatial import ResNet18Spatial, ResNet34Spatial
import Record as rc
import sys

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Global Config
# ----------------------------------------------------------------------------------------------------------------------
# Global Adjustments:
NETS = [ResNet18, ResNet18Spatial, ResNet34Spatial]
NET = NETS[1]  # The c
BATCH_SIZE = 128

# Verbosity Adjustments:
VERBOSITY = 1  # 0 for per epoch output, 1 for per-batch output

# ----------------------------------------------------------------------------------------------------------------------
#                                                Optimization Functionality
# ----------------------------------------------------------------------------------------------------------------------
TEST_SET_SIZE = 1000  # Max for CIFAR10 is 10000
MAX_POSSIBILITIES = 10001
MAX_ACC_LOSS = 1.83

PS = 2  # TODO - Check with other patch sizes (resnet_spatial implementation was changed)
ONES_RANGE = (1, 3)  # Exclusive range

GRAN_THRESH = 32 * 32  # 32*32 will do nothing. The smaller this is, the bigger the effect per patch we expect

# Complexity Config
SAVE_INTERVAL = 300
# ----------------------------------------------------------------------------------------------------------------------
#                                                   Data Import
# ----------------------------------------------------------------------------------------------------------------------
if 'google.colab' in sys.modules:
    CHECKPOINT_DIR = '/content/drive/My Drive/Colab Notebooks/data/checkpoint/'
    RESULTS_DIR = '/content/drive/My Drive/Colab Notebooks/data/results/'
    DO_DOWNLOAD = True
else:
    CHECKPOINT_DIR = './data/checkpoint/'
    RESULTS_DIR = './data/results/'
    DO_DOWNLOAD = False
# ----------------------------------------------------------------------------------------------------------------------
#                                                Train Specific Functionality
# ----------------------------------------------------------------------------------------------------------------------
# Checkpoint Adjustments
RESUME_CHECKPOINT = True
RESUME_METHOD = 'ValAcc'  # 'ValAcc' 'Time'

# Train Adjustments - Notice that there is currently no learning step decrease functionality
N_EPOCHS = 30
TRAIN_SET_SIZE = 5000  # Max 50000 for CIFAR10
LEARN_RATE = 0.1
DONT_SAVE_REDUNDANT = True  # Don't checkpoint if val_acc achieved is lower than what is in the cp directory
