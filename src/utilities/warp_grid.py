
from scipy import interpolate
import numpy as np
import cv2

def warp_matrix_old1(matrix, offset):
    """

    :param matrix: np.array(float) (Height,Width)
    :param coords: np.array(float) (YX,Height,Width)
    :return: np.array(float) (Height,Width)
    """
    height = matrix.shape[0]
    width = matrix.shape[1]

    x_original_coords, y_original_coords = np.meshgrid(np.arange(width), np.arange(height))

    x_warped_coords = x_original_coords + offset[1]
    y_warped_coords = y_original_coords + offset[0]

    interpolate_func = interpolate.interp2d(np.arange(width),np.arange(height),matrix)

    interpolated_grid = np.array([interpolate_func(x,y) for x,y in zip(x_warped_coords.flatten(),y_original_coords.flatten())])
    interpolated_grid.shape = (height,width)
    return interpolated_grid


def warp_matrix_old2(matrix, offset):
    """

    :param matrix: np.array(float) (Height,Width)
    :param coords: np.array(float) (YX,Height,Width)
    :return: np.array(float) (Height,Width)
    """
    height = matrix.shape[0]
    width = matrix.shape[1]

    x_original_coords, y_original_coords = np.meshgrid(np.arange(width), np.arange(height))

    x_warped_coords = x_original_coords + offset[1]
    y_warped_coords = y_original_coords + offset[0]

    interpolate_func = interpolate.RectBivariateSpline(np.arange(height),np.arange(width),matrix)

    interpolated_grid = interpolate_func.ev(y_warped_coords.flatten(),x_warped_coords.flatten())
    interpolated_grid.shape = (height,width)
    return interpolated_grid

def warp_matrix(matrix, offset):
    """

    :param matrix: np.array(float) (Height,Width)
    :param coords: np.array(float) (YX,Height,Width)
    :return: np.array(float) (Height,Width)
    """
    height = matrix.shape[0]
    width = matrix.shape[1]
    matrix = np.array(matrix,dtype=np.float32,copy=False)
    x_original_coords, y_original_coords = np.meshgrid(np.arange(width), np.arange(height))

    x_warped_coords = x_original_coords + offset[1]
    y_warped_coords = y_original_coords + offset[0]
    x_warped_coords = x_warped_coords.astype(np.float32)
    y_warped_coords = y_warped_coords.astype(np.float32)

    return cv2.remap(matrix,x_warped_coords,y_warped_coords,cv2.INTER_CUBIC)

def warp_image(image,warp_offset):
    """

    :param image: np.array(float) (ColorChannels,Height,Width)
    :param warp_offset: np.array(float) (YX,Height,Width)
    :return: np.array(float) (ColorChannels,Height,Width)
    """
    color_channels_count = image.shape[0]
    warped_img = np.empty(shape=image.shape,dtype=float)

    for channel_idx in range(color_channels_count):
        warped_img[channel_idx]=warp_matrix(image[channel_idx],warp_offset)
    return warped_img

def warp_derivative(derivative,warp_offset):
    """

    :param derivative: np.array(float) (ColorChannels,YX,Height,Width)
    :param warp_offset: np.array(float) (YX,Height,Width)
    :return: (ColorChannels,YX,Height,Width)
    """
    warped_derivative = np.empty(shape=derivative.shape, dtype=float)
    color_channels_count = derivative.shape[0]
    for channel_idx in range(color_channels_count):
        warped_derivative[channel_idx] = warp_image(derivative[channel_idx], warp_offset)
    return warped_derivative