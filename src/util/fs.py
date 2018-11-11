import os


def enable_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
