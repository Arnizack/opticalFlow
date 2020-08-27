from src.utilities.warp_grid import warp_image,warp_derivative
from src.utilities.image_access import open_image,show_image
from src.horn_schunck.derivative_horn_schunck import get_I_x,get_I_y,get_I_t
from src.utilities.image_derivative import differentiate_matrix
from scipy import sparse
import scipy.sparse.linalg as splinalg
import numpy as np
import matplotlib.pyplot as plt

from time import time


def setup_outer_array(width,height,value):
    corner_convolution = np.zeros(shape=(width * height * 2))

    temp_corner_convolution = np.full(shape=(height), fill_value=value)
    temp_corner_convolution[-1] = 0
    temp_corner_convolution = np.repeat([temp_corner_convolution], width-1, axis=0).flatten()

    corner_convolution[0:height * (width-1)] = temp_corner_convolution
    corner_convolution[width * height:-height] = temp_corner_convolution

    return corner_convolution

def setup_inner_array(width,height,value):
    side_array = np.zeros(shape=(width * height * 2))
    temp_side_array = np.full(shape=(height),fill_value=value)
    temp_side_array[-1] = 0
    side_array = np.repeat([temp_side_array], width*2 , axis=0).flatten()

    return side_array


def offset_array(np_array,offset):
    """

    :param np_array: numpy array 1D
    :param offset: postiv int
    :return: numpy array
    """
    if(offset>=1):
        result = np.zeros(shape=np_array.shape,dtype=np_array.dtype)
        result[offset:]=np_array[:-offset]
        return result
    return np_array

def setup_diagonals_convolution(width,height,alpha):
    offset = [height,height+1,height-1,
              1,-1,
              -height+1,-height,-height-1]
    corner_val = -alpha**2/12
    side_val = -alpha**2/6
    corner_upper_left = setup_outer_array(width,height,corner_val)
    corner_upper_left = offset_array(corner_upper_left,height)
    corner_upper_right = setup_outer_array(width,height,corner_val)
    corner_upper_right = offset_array(corner_upper_right,height+1)

    side_outer_array_up = np.zeros(shape=(width*height*2))
    side_outer_array_up[:width*height]=np.full(shape=(width*height),fill_value=side_val)
    side_outer_array_up[(width+1)*height:] = np.full(shape=((width-1)*height),fill_value=side_val)

    side_inner_array_right = setup_inner_array(width,height,side_val)
    side_inner_array_right = offset_array(side_inner_array_right,1)

    side_inner_array_left = setup_inner_array(width,height,side_val)



    corner_down_right = setup_outer_array(width,height,corner_val)
    corner_down_left = setup_outer_array(width, height, corner_val)
    corner_down_left = offset_array(corner_down_left,1)

    side_outer_array_down= np.zeros(shape=(width * height * 2))
    side_outer_array_down[:(width-1) * height] = np.full(shape=((width-1) * height), fill_value=side_val)
    side_outer_array_down[width  * height:] = np.full(shape=(width * height), fill_value=side_val)

    diagonals = [side_outer_array_up,corner_upper_right,corner_upper_left,
                 side_inner_array_right,side_inner_array_left,
                 corner_down_left,side_outer_array_down,corner_down_right]
    return (diagonals,offset)

def setup_diagonals_image(width,height,I_xx,I_xy,I_yy,alpha):
    middle = np.empty(shape=(width*height*2))
    middle[:width*height]=I_yy.flatten()
    middle[width*height:]=I_xx.flatten()
    R = np.empty(shape=(width*height*2))
    R[:width * height] = I_xy.flatten()
    R[width * height:] = I_xy.flatten()

    middle+=alpha**2

    offset = [-width*height,0,width*height]
    diags = [R,middle,R]
    return (diags,offset)

def setup_linear_system(first_image,second_image,first_image_derivative,second_image_derivative,alpha):

    channel_count = first_image.shape[0]
    height = first_image.shape[2] #switched height,width
    width = first_image.shape[1]

    # see Horn–Schunck Optical Flow with a Multi-Scale Strategy page 4 for reference

    I_xx = np.zeros(shape=(height * width), dtype=float)
    I_xy = np.zeros(shape=(height * width), dtype=float)
    I_yy = np.zeros(shape=(height * width), dtype=float)
    I_xt = np.zeros(shape=(height * width), dtype=float)
    I_yt = np.zeros(shape=(height * width), dtype=float)
    for channel_idx in range(channel_count):

        first_yx = differentiate_matrix(first_image[channel_idx])
        channel_first_y,channel_first_x = first_yx
        channel_first_x = channel_first_x.flatten()
        channel_first_y = channel_first_y.flatten()

        channel_second_y, channel_second_x = differentiate_matrix(second_image[channel_idx])
        channel_second_x = channel_second_x.flatten()
        channel_second_y = channel_second_y.flatten()

        """
        channel_first_x = first_image_derivative[channel_idx][1].flatten()
        channel_first_y = first_image_derivative[channel_idx][0].flatten()

        channel_second_x = second_image_derivative[channel_idx][1].flatten()
        channel_second_y = second_image_derivative[channel_idx][0].flatten()
        """
        #deriviative blending: see An Improved Algorithm for TV-L1 Optical Flow
        b=0.4
        channel_x = channel_first_x*b+channel_second_x*(1-b)
        channel_y = channel_first_y*b+channel_second_y*(1-b)

        """
        print("channel_x, channel_y")
        plt.figure()
        plt.imshow(channel_x.reshape(height,width))
        plt.figure()
        plt.imshow(channel_y.reshape(height,width))
        plt.show()
        """

        channel_t = second_image[channel_idx] - first_image[channel_idx]
        channel_t = channel_t.flatten()

        channel_x = get_I_x(first_image[channel_idx],second_image[channel_idx]).flatten()
        channel_y = get_I_y(first_image[channel_idx], second_image[channel_idx]).flatten()
        channel_t = get_I_t(first_image[channel_idx], second_image[channel_idx]).flatten()


        I_xx += channel_x ** 2
        I_xy += channel_x * channel_y
        I_yy += channel_y ** 2
        I_xt += channel_x * channel_t
        I_yt += channel_y * channel_t

    channel_count_squared = channel_count**2

    I_xx/=channel_count_squared
    I_xy /= channel_count_squared
    I_yy /= channel_count_squared
    I_xt /= channel_count_squared
    I_yt /= channel_count_squared
    """
    print("I_xx, I_xy, I_yy, I_xt, I_yt")
    plt.figure()
    plt.imshow(I_xx.reshape(width,height))
    plt.figure()
    plt.imshow(I_xy.reshape(width,height))
    plt.figure()
    plt.imshow(I_yy.reshape(width,height))
    plt.show()"""

    diags1,offset1 = setup_diagonals_image(width,height,I_xx,I_xy,I_yy,alpha)

    diags2,offset2 = setup_diagonals_convolution(width,height,alpha)

    diags = diags1+diags2
    offset = offset1+offset2

    A = sparse.spdiags(diags,offset,width*height*2,width*height*2)
    b = np.zeros(shape=(width*height*2))
    b[:width * height] = -I_yt.flatten()
    b[width * height:] = -I_xt.flatten()

    return (A,b)