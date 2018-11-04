from inspect import currentframe, getframeinfo
from datetime import datetime

# ----------------------------------------------------------------------------------------------------------------------
#                                               	    Misc
# ----------------------------------------------------------------------------------------------------------------------

def banner(text, ch='=', length=78):
    spaced_text = ' %s ' % text
    print(spaced_text.center(length, ch))

def format_seconds(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


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

