import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def grayScaleGradient(np_img,direction):
    """
    Bekommt einen 2D numpy Array und erstellt von diesem die Ableitung
    direction kann entweder X oder Y sein
    """
    kernal = np.array(
        [[1,0,-1],
        [2,0,-2],
        [1,0,-1],])/3
    kernal = np.array(
        [[-1,8,0,-8,1]]
    )*1/12
    if(direction=="X"):
        kernal = kernal.T

    np_grad = signal.convolve2d(np_img,kernal, boundary="symm", mode = "same")

    return np_grad