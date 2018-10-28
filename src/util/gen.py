from inspect import currentframe, getframeinfo
from datetime import datetime
import numpy as np
import sys

# ----------------------------------------------------------------------------------------------------------------------
#                                               	    Misc
# ----------------------------------------------------------------------------------------------------------------------

def banner(text, ch='=', length=78):
    spaced_text = ' %s ' % text
    print(spaced_text.center(length, ch))

# ----------------------------------------------------------------------------------------------------------------------
#                                                   Model Test Suite
# ----------------------------------------------------------------------------------------------------------------------

def assert_eq(tested, target, desc=""):
    if (tested != target):
        f = currentframe().f_back
        print('[{} :: {} :: line {} ] Test Failure - {}\nTested = {}\nTarget = {}\nExiting due to failed test'.format(
            datetime.now().strftime('%H:%M:%S'), getframeinfo(f).filename.split('/')[-1], getframeinfo(f).lineno, desc,
            tested, target))
        exit(-1)

