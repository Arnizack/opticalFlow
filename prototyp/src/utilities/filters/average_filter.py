import numpy as np
from scipy.signal import convolve2d

def filter_average(matrix2d,size=3,direction="XY"):

    if(direction=="XY"):
        kernel_shape = (size,size)
    elif (direction=="X"):
        kernel_shape = (1,size)
    else:
        kernel_shape = (size,1)

    kernel = np.full(fill_value=1/size,shape=kernel_shape)
    return convolve2d(matrix2d,kernel,mode="same", boundary='symm')