import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse

from util.meta import Singleton


class RunDefaults(metaclass=Singleton):
    def __init__(self):
        # Set your desired global defaults here
        self._ALLOW_CPU = False
        self._ROOT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self._CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'checkpoint')
        self._DEFAULT_LR = 0.1
        self._PRESUME_RESUME  = False


    def device(self):
        if torch.cuda.is_available():
            return 'cuda'
        elif self._ALLOW_CPU:
            return 'cpu'
        else:
            raise AssertionError('Fatal: Did not find a valid NVIDIA GPU, and ALLOW_CPU flag is off')

    @property
    def data_path(self):
        return self._ROOT_DATA_PATH

    @property
    def checkpoint_path(self):
        return self._CHECKPOINT_PATH
    @property
    def def_lr(self):
        return self._DEFAULT_LR
    @property
    def do_resume(self):
        return self._PRESUME_RESUME



class RunConfig(metaclass=Singleton):
    def __init__(self,nn_creator,data_importer,cp_target='LATEST', criterion=nn.CrossEntropyLoss()):
        self.args =  self._command_line_parse()
        self.defs = RunDefaults()

        # Construct the Run Configuration:
        self.nn_creator = nn_creator
        self.data_importer = data_importer
        self.crit = criterion

        # Calculate the device needed (Default config only)
        self.device = self.defs.device()

        self.cp_target = _calc_resume_target(cp_target):


        start_epoch # start from epoch 0 or last checkpoint epoch

    def resume(self):
        if not args or args.resume == False or self.resume_target


    def __str__(self):
        pass


    # Private functions
    def _calc_resume_target(self):
        if is

    def _command_line_parse(self):
        parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
        parser.add_argument('--lr', default=self.defs.def_lr(), type=float, help='learning rate')
        parser.add_argument('--resume', '-r', default=self.defs.do_resume(), help='checkpoint file')
        return parser.parse_args()  # Not sure how this acts on empty string


list_of_files = glob.glob('/path/to/folder/*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)