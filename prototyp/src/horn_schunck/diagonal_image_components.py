from src.utilities.warp_grid import warp_image,warp_derivative
from src.utilities.image_access import open_image,show_image
from scipy import sparse
import scipy.sparse.linalg as splinalg

from src.horn_schunck.derivative_horn_schunck import get_I_x,get_I_y,get_I_t
from src.utilities.image_derivative import differentiate_matrix
from src.utilities.filters.average_filter import filter_average

import numpy as np
import matplotlib.pyplot as plt


def convert_img_to_gray(img):
    channel_count = img.shape[0]
    height = img.shape[1]
    width = img.shape[2]
    gray_img = np.zeros(shape=(height,width),dtype=float)
    for channel_idx in range(channel_count):
        gray_img+=img[channel_idx]
    gray_img/channel_count
    return gray_img

def derivative_horn_schunck(gray_first_image,gray_second_image):
    I_x = get_I_x(gray_first_image,gray_second_image)
    I_y = get_I_y(gray_first_image,gray_second_image)
    I_t = get_I_t(gray_first_image,gray_second_image)


    return I_x,I_y,I_t


def derivative_sun(gray_first_image,gray_second_image):
    #See: A Quantitative Analysis of Current Practices in Optical Flow Estimation
    #and the Principles behind Them
    first_I_y,first_I_x = differentiate_matrix(gray_first_image)
    second_I_y, second_I_x = differentiate_matrix(gray_second_image)
    b=0.4

    I_x = first_I_x * b + second_I_x * (1-b)
    I_y = first_I_y * b + second_I_y * (1-b)
    I_t = gray_second_image-gray_first_image


    return I_x,I_y,I_t



def get_img_diags(first_image, second_image,derivative_typ):
    gray_first_image = convert_img_to_gray(first_image)
    gray_second_image = convert_img_to_gray(second_image)

    derivative_mode = derivative_typ
    if(derivative_mode=="Horn-Schunck"):
        I_x,I_y,I_t = derivative_horn_schunck(gray_first_image,gray_second_image)
    else:
        I_x, I_y, I_t = derivative_sun(gray_first_image,gray_second_image)

    I_x = I_x.flatten()
    I_y = I_y.flatten()
    I_t = I_t.flatten()

    I_xx = I_x**2
    I_xy = I_x * I_y
    I_yy = I_y**2
    I_xt = I_x * I_t
    I_yt = I_y * I_t

    return (I_xx,I_xy,I_yy,I_xt,I_yt)


def get_img_diags_color(first_image, second_image,derivative_typ):
    channel_count = first_image.shape[0]
    height = first_image.shape[1]
    width = first_image.shape[2]
    I_xx = np.zeros(shape=(height, width), dtype=float)
    I_xy = np.zeros(shape=(height, width), dtype=float)
    I_yy = np.zeros(shape=(height, width), dtype=float)
    I_xt = np.zeros(shape=(height, width), dtype=float)
    I_yt = np.zeros(shape=(height, width), dtype=float)

    derivative_mode = derivative_typ

    for channel_idx in range(channel_count):

        if (derivative_mode == "Horn-Schunck"):
            I_x, I_y, I_t = derivative_horn_schunck(first_image[channel_idx], second_image[channel_idx])
        else:
            I_x, I_y, I_t = derivative_sun(first_image[channel_idx], second_image[channel_idx])

        I_xx += I_x ** 2
        I_xy += I_x * I_y
        I_yy += I_y ** 2
        I_xt += I_x * I_t
        I_yt += I_y * I_t

    I_x = I_x.flatten()
    I_y = I_y.flatten()
    I_t = I_t.flatten()

    return (I_xx,I_xy,I_yy,I_xt,I_yt)
