from scipy.misc import imread
import PIL
from PIL import Image

# ----------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------

def imread(filename, mode='L',raw=False):
    """
    Reads an image from a file
    :param filename: string
    :param mode: string: 'L' for grayscale or 'RGB' for RGB
    :param raw: boolean: If True, returns an uint8 numpy.ndarray and not a PIL.Image object
    :return: Either a PIL.Image object or a uint8 numpy.ndarray as the image
    """
    if raw:
        return imread(filename,mode)
    else:
        return Image.open(filename).convert(mode)

def imshow(im,title=None):
    """
    Shows the image im with title title
    :param im: PIL.Image|uint8 numpy.ndarray : The image
    :param title: string : Optional - Does not work for some image viewers
    """
    if not isinstance(im,PIL.Image.Image):
        im = Image.fromarray(im)

    print("Image Size: {} x {} of mode: {}".format(im.height,im.width,im.mode))
    im.show(title)

