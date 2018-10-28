from inspect import currentframe, getframeinfo
from datetime import datetime
import numpy as np
import sys

# ----------------------------------------------------------------------------------------------------------------------
#                                               	Data Searching
# ----------------------------------------------------------------------------------------------------------------------

def sub2ind(t, i, j):
    return i * t + j

def ind2sub(t, ind):
    return (ind // t, ind % t)

# ----------------------------------------------------------------------------------------------------------------------
#                                               	Data Handling
# ----------------------------------------------------------------------------------------------------------------------

def shuffle(x, y):
    """
    :param x: numpy array, sample or samples, t^2 X num_features each
    :param y: numpy array, label or labels, can be one-hot (t^2 X t^2 each) or integers (t^2 each)
    :return: numpy arrays of shuffled sample(s) and label(s)
    """
    length = x.shape[0]
    perm = np.random.permutation(length)
    x_s, y_s = [], []
    [x_s.append(x[i]) for i in perm]
    [y_s.append(y[i]) for i in perm]

    return np.array(x_s), np.array(y_s)

# ----------------------------------------------------------------------------------------------------------------------
#                                               	Data Download & Read
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


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    # Uses TensorFlow as tf
    # Usage: 
    #   vocabulary = read_data(filename)
    #   print(vocabulary[:7])
    #   ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse']
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


