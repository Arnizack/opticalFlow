import scipy.ndimage as ndimage
from math import *
import numpy as np

def downscale(matrix2d,factor):
    # sigma same as in A Quantitative Analysis of Current Practices in Optical Flow Estimation
    # and the Principles behind Them; Deqing Sun Stefan Roth Michael J. Black
    sigma = 1 / sqrt(2 * factor)
    np_blured = ndimage.gaussian_filter(matrix2d, sigma)
    #mode = nearest interpolation
    return ndimage.zoom(np_blured,factor,order=0)

def downscale_image(np_img,factor):
    img_channels = []
    for channel in range(np_img.shape[0]):
        img_channel = downscale(np_img[channel],factor)
        img_channels.append(img_channel)
    return np.array(img_channels,copy=False,dtype=float)

def create_image_pyramid(np_img,factors):
    """
    :param np_img: (ColorSpace,Height,Width)
    :param factors:  Array of the downscale factors. They should be between 0 and 1
    :return: Array of the Images (smallest first)
    """
    pyramid_levels = [np_img]
    for factor in factors:
        pyramid_levels.append(downscale_image(pyramid_levels[-1],factor))

    pyramid_levels.reverse()
    return pyramid_levels