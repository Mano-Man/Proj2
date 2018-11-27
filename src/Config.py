import torch
# Models: (There are many more)
from models.resnet import ResNet18
from models.resnet_spatial import ResNet18Spatial, ResNet34Spatial
import Record as rc

# ----------------------------------------------------------------------------------------------------------------------
#                                                 Global Config
# ----------------------------------------------------------------------------------------------------------------------
# Global Adjustments:
NETS = [ResNet18, ResNet18Spatial, ResNet34Spatial]
NET = NETS[1]
BATCH_SIZE = 128

# Verbosity Adjustments:
VERBOSITY = 1  # 0 for per epoch output, 1 for per-batch output
DO_DOWNLOAD = True

# Checkpoint Adjustments
RESUME_CHECKPOINT = True
RESUME_METHOD = 'ValAcc'  # 'ValAcc' 'Time'
CHECKPOINT_DIR = '/content/drive/My Drive/Colab Notebooks/data/checkpoint/'
RESULTS_DIR =  '/content/drive/My Drive/Colab Notebooks/data/results/'
# ----------------------------------------------------------------------------------------------------------------------
#                                               Spatial Functionality
# ----------------------------------------------------------------------------------------------------------------------
# Spatial Config
RECORDS_FILENAME = 'ps2_ones(1, 3)_uniform_patch_acc93.83_mg1024_17860D15h'
GEN_PATTERNS = True
ONES_RANGE = (1, 3)  # Exclusive range

MODE = rc.uniform_filters
LAYER_LAYOUT = rc.Resnet18_layers_layout
PS = 2
GRAN_THRESH = 32 * 32

# Complexity Config
SAVE_INTERVAL = 100
RESUME_MASK_GEN = False

# Dummy workloads
SP_MOCK = [(0, 0, 0)] * len(LAYER_LAYOUT)
SP_ZEROS = [(1, PS, torch.zeros([64, 32, 32])),
            (1, PS, torch.zeros([64, 32, 32])),
            (1, PS, torch.zeros([128, 16, 16])),
            (1, PS, torch.zeros([256, 8, 8])),
            (1, PS, torch.zeros([512, 4, 4]))]

CHOSEN_SP = SP_MOCK
# ----------------------------------------------------------------------------------------------------------------------
#                                                Train Functionality
# ----------------------------------------------------------------------------------------------------------------------
# Train Adjustments - TODO - Write a learning step decrease functionality
N_EPOCHS = 30
TRAIN_SET_SIZE = 50000  # Max 50000 for CIFAR10
LEARN_RATE = 0.1
DONT_SAVE_REDUNDANT = True  # Don't checkpoint if val_acc achieved is lower than what is in the cp directory
