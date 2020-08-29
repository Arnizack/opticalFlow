from scipy.ndimage.filters import gaussian_filter,median_filter
from scipy.signal import convolve2d
import numpy as np

def differentiate_matrix(matrix2d):
    """

    :param matrix2d:
    :return: Array with Shape (YX,Height,Width)
    """
    # Same Kernel as in A Quantitative Analysis of Current Practices in Optical Flow Estimation
    # and the Principles behind Them; Deqing Sun Stefan Roth Michael J. Black
    #kernel = 1/12 * np.array([-1,8,0,-8,1])
    kernel_x =  np.array([[-1, 8, 0, -8, 1]])/12
    kernel_y = np.array([[-1], [8], [0], [-8], [1]])/12
    y_derivative = convolve2d(matrix2d,kernel_y,mode="same", boundary='symm')
    x_derivative = convolve2d(matrix2d, kernel_x,mode="same", boundary='symm')
    #y_derivative=median_filter(y_derivative,size=3)
    # = median_filter(x_derivative, size=3)
    return np.array([y_derivative,x_derivative],copy=False)

def differentiate_image(np_img):
    """

    :param np_img: numpy
    :return: Array  with shape (ColorChannels,YX,Height,Width)
    """
    color_channels_count = np_img.shape[0]
    channels = []
    for channel_idx in range(color_channels_count):
        channels.append(differentiate_matrix(np_img[channel_idx]))
    return np.array(channels,copy=False)

def differentiate_images(np_images):
    """

    :param np_img: numpy
    :return: Array  with shape (np_images_count,ColorChannels,YX,Height,Width)
    """
    return [differentiate_image(img) for img in np_images]