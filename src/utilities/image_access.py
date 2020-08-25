from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

"""
Image Coordinate Space:
(ColorSpace, Width, Heigth)
"""

def _WHC_to_CWH_space(np_img):
    """
    CWH Space = (ColorSpace, Width, Heigth)
    :param np_img:
    :return:
    """
    ColorSpaceCount = np_img.shape[2]
    return np.array([ np_img[:,:,i].astype(float)/255 for i in range(ColorSpaceCount)],dtype=float)

def _CWH_to_WHC_space(np_img):
    ColorSpaceCount = np_img.shape[0]
    feld = np.empty(shape=[np_img.shape[1],np_img.shape[2],ColorSpaceCount],dtype=np.uint8)
    for i in range(ColorSpaceCount):
        feld[:,:,i]=np_img[i]*255
    return feld

def open_image(filepath):
    pil_img = Image.open(filepath)
    np_img = np.asarray(pil_img)

    return _WHC_to_CWH_space(np_img)


def save_image(filepath, np_img):
    np_img_WHC = _CWH_to_WHC_space(np_img)
    pil_img = Image.fromarray(np_img_WHC)
    pil_img.save(filepath)

def show_image(np_img):
    np_img_WHC = _CWH_to_WHC_space(np_img)
    pil_img = Image.fromarray(np_img_WHC)
    plt.imshow(pil_img)