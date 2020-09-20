import numpy as np
from scipy import sparse
from typing import List, Dict


def zero_padding_diags(value, width, height):
    temp_corner_convolution = np.full(shape=(width), fill_value=value)
    temp_corner_convolution[-1] = 0
    return np.repeat([temp_corner_convolution], height, axis=0).flatten()


def offset_array(array, offset):
    result = np.zeros(shape=(array.shape[0]))
    result[offset:] = array[:-offset]
    return result


def fill_at_position(array, fill_value, position, width):
    array[position:position + width] = fill_value


def convolution3x3(kernel: List[List[float]], width: int, height: int, padding="zero") -> sparse.dia_matrix:
    """
    Creates the diagonals, from which an diagonal Matrix A can be construced.
    The matrix multiplication Ax is the same as the convolution on an image.
    x is the image in 1D shape with row major order

    :param kernel: (3,3) float
    :param width: int
    :param height: int
    :return: Dict, where the keys are the offsets and the value are the diagonals
    """
    """
    kernel = [[1,2,3],
              [4,5,6],
              [7,8,9]]
    
    width = 3
    height = 4
    
    |x| = 3*4=12
    
    A =
    5, 6, 0, | 8, 9, 0, | 0, 0, 0, | 0, 0, 0  
    4, 5, 6, | 7, 8, 9, | 0, 0, 0, | 0, 0, 0
    0, 4, 5, | 0, 7, 8, | 0, 0, 0, | 0, 0, 0
    
    2, 3, 0, | 5, 6, 0, | 8, 9, 0, | 0, 0, 0
    1, 2, 3, | 4, 5, 6, | 7, 8, 9, | 0, 0, 0
    0, 1, 2, | 0, 4, 5, | 0, 7, 8, | 0, 0, 0
    
    0, 0, 0, | 2, 3, 0, | 5, 6, 0, | 8, 9, 0 
    0, 0, 0, | 1, 2, 3, | 4, 5, 6, | 7, 8, 9 
    0, 0, 0, | 0, 1, 2, | 0, 4, 5, | 0, 7, 8 
    
    0, 0, 0, | 0, 0, 0, | 2, 3, 0, | 5, 6, 0
    0, 0, 0, | 0, 0, 0, | 1, 2, 3, | 4, 5, 6
    0, 0, 0, | 0, 0, 0, | 0, 1, 2, | 0, 4, 5
        
    """

    diags_dict = {}

    diags_dict[-1] = zero_padding_diags(kernel[1][0], width, height)
    diags_dict[0] = np.full(shape=(width * height), fill_value=kernel[1][1])
    diags_dict[1] = zero_padding_diags(kernel[1][2], width, height)

    diags_dict[1] = offset_array(diags_dict[1], 1)

    diags_dict[width - 1] = zero_padding_diags(kernel[2][0], width, height)
    diags_dict[width] = np.full(shape=(width * height), fill_value=kernel[2][1])
    diags_dict[width + 1] = zero_padding_diags(kernel[2][2], width, height)

    diags_dict[width - 1] = offset_array(diags_dict[width - 1], width)
    diags_dict[width] = offset_array(diags_dict[width], width)
    diags_dict[width + 1] = offset_array(diags_dict[width + 1], width + 1)

    diags_dict[-width + 1] = zero_padding_diags(kernel[0][2], width, height)
    diags_dict[-width] = np.full(shape=(width * height), fill_value=kernel[0][1])
    diags_dict[-width - 1] = zero_padding_diags(kernel[0][0], width, height)

    diags_dict[-width + 1] = offset_array(diags_dict[-width + 1], 1)

    fill_at_position(diags_dict[-width + 1], 0, (height - 1) * width, width)
    fill_at_position(diags_dict[-width - 1], 0, (height - 1) * width, width)
    fill_at_position(diags_dict[-width], 0, (height - 1) * width, width)

    diags = list(diags_dict.values())
    offsets = list(diags_dict.keys())

    return sparse.spdiags(diags,offsets,width*height,width*height)
