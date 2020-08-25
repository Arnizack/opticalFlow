from scipy.ndimage.filters import convolve1d
import numpy as np

def differentiate_matrix(matrix2d):
    """

    :param matrix2d:
    :return: Array with Shape (2,Width,Height)
    """
    # Same Kernel as in A Quantitative Analysis of Current Practices in Optical Flow Estimation
    # and the Principles behind Them; Deqing Sun Stefan Roth Michael J. Black
    kernel = 1/12 * np.array([-1,8,0,-8,1])
    x_derivative = convolve1d(matrix2d,kernel,axis=0)
    y_derivative = convolve1d(matrix2d, kernel, axis=1)
    return np.array([x_derivative,y_derivative],copy=False)

def differentiate_image(np_img):
    """

    :param np_img: numpy
    :return: Array  with shape (ColorChannels,Width,Height)
    """
    color_channels_count = np_img.shape[0]
    channels = []
    for channel_idx in range(color_channels_count):
        channels.append(differentiate_matrix(np_img[channel_idx]))
    return np.array(channels,copy=False)

def differentiate_images(np_images):
    """

    :param np_img: numpy
    :return: Array  with shape (np_images_count,ColorChannels,Width,Height)
    """
    return [differentiate_image(img) for img in np_images]