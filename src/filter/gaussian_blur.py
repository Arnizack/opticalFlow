import numpy as np
from scipy import signal
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import math

def gaussian_kern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern1d/= gkern1d.sum()
    np.array([[1, 1, 1]]) / 3
    return gkern1d


def gaussian_blur_matrix(matrix, std=3):

    return filters.gaussian_filter(matrix,std)

def gaussian_blur_flow(flow,std=3):

    outofrange_y = np.abs(flow[0]) > flow.shape[1]
    flow[0][outofrange_y] = 0

    outofrange_x = np.abs(flow[1]) > flow.shape[2]
    flow[1][outofrange_x] = 0

    flow_x = filters.gaussian_filter(flow[1],std)
    flow_y = filters.gaussian_filter(flow[0],std)



    return np.array([flow_y,flow_x])
