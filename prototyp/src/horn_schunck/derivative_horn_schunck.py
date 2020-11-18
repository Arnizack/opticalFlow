import numpy as np


def add_with_offset(dst,src,offset_y,offset_x):
    """

    :param dst: 2d np.array
    :param src: 2d np.array
    :param offset_x: positiv int
    :param offset_y: positiv int
    :return: None
    """
    src_width = src.shape[1]
    src_height= src.shape[0]
    src_shifted = src[offset_y:,offset_x:]
    dst[:src_height-offset_y,:src_width-offset_x]+=src_shifted
    pass


def get_I_x(img1,img2):
    """

    :param img1: 2D np.array
    :param img2: 2D np.array
    :return: 2D np.array
    """
    width = img1.shape[1]
    height = img1.shape[0]

    temp_I_x = np.zeros(shape=(height+1,width+1))
    for img in [img1,img2]:
        add_with_offset(temp_I_x,img,0,1)
        add_with_offset(temp_I_x,-img,0,0)
        add_with_offset(temp_I_x,img,1,1)
        add_with_offset(temp_I_x,-img,1,0)
    I_x = np.zeros(shape=(height,width))
    I_x[:height-1,:width-1] = temp_I_x[:height-1,:width-1]
    return I_x/4

def get_I_y(img1,img2):
    """

    :param img1: 2D np.array
    :param img2: 2D np.array
    :return: 2D np.array
    """
    width = img1.shape[1]
    height = img1.shape[0]

    temp_I_y = np.zeros(shape=(height+1,width+1))
    for img in [img1,img2]:
        add_with_offset(temp_I_y,img,1,0)
        add_with_offset(temp_I_y,-img,0,0)
        add_with_offset(temp_I_y,img,1,1)
        add_with_offset(temp_I_y,-img,0,1)
    I_y = np.zeros(shape=(height,width))
    I_y[:height-1,:width-1] = temp_I_y[:height-1,:width-1]
    return I_y/4

def get_I_t(img1,img2):
    """

    :param img1: 2D np.array
    :param img2: 2D np.array
    :return: 2D np.array
    """
    width = img1.shape[1]
    height = img1.shape[0]

    temp_I_t = np.zeros(shape=(height+1,width+1))

    add_with_offset(temp_I_t,img2,0,0)
    add_with_offset(temp_I_t, -img1, 0, 0)
    add_with_offset(temp_I_t, img2, 1, 0)
    add_with_offset(temp_I_t, -img1, 1, 0)
    add_with_offset(temp_I_t, img2, 0, 1)
    add_with_offset(temp_I_t, -img1, 0, 1)
    add_with_offset(temp_I_t, img2, 1, 1)
    add_with_offset(temp_I_t, -img1, 1, 1)

    I_t = np.zeros(shape=(height,width))
    I_t[:height-1,:width-1] = temp_I_t[:height-1,:width-1]
    return I_t/4
