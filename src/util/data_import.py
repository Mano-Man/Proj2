import os
import urllib


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


