import numpy as np
import scipy
from scipy import interpolate
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import logging


class Interpolator:
    x_min=0
    x_max=0
    y_min=0
    y_max=0
    def __init__(self,inter2dObj):
        self.inter2dObj = inter2dObj
        self.x_min=inter2dObj.x_min
        self.x_max=inter2dObj.x_max
        self.y_min=inter2dObj.y_min
        self.y_max=inter2dObj.y_max
    def __call__(self,x,y):
        return self.inter2dObj(y,x)

def interpolate2dArray(np_array):
    """
    np_array ist ein 2D numpy Array 
    return Interpolator Obj des Arrays
    """
    #Sicherheit
    if(type(np_array)!=np.ndarray):
        raise Exception("np_array muss ein numpy Array sein")
    if(len(np_array.shape)!=2):
        raise Exception("np_array muss ein 2D numpy Array sein")

    x = np.arange(np_array.shape[0])
    y = np.arange(np_array.shape[1])


    inter2dObj = interpolate.interp2d(y,x,np_array,fill_value=0,kind="cubic")
    result = Interpolator(inter2dObj)
    return result

