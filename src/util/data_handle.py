import numpy as np


def sub2ind(t, i, j):
    return i * t + j

def ind2sub(t, ind):
    return (ind // t, ind % t)

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

