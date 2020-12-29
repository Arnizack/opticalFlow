import numpy as np
from scipy import interpolate
import numpy as np
import cv2
from utilities import warp_grid
import math

def shift_array(array,x,y):
    array = np.roll(array,y,axis=0)
    array = np.roll(array,x,axis=1)
    #zero padding
    if(y < 0):
        array[y:len(array.shape[0])] = 0
    else:
        array[0:y] = 0
    
    if(x < 0):
        array[:,x:len(array.shape[1])] = 0
    else:
        array[:,0:x] = 0
    
    return array

def sun_baker_error_map(image_first, image_second,flow, lambda_derivative, lambda_flow, penalty_direct, 
                             penalty_spatial, derivative_x_func,derivative_y_func, neighboor_size):
    flow_x = flow[1]
    flow_y = flow[0]
    image_second_warped = warp_grid.warp_matrix(image_second,flow)
    error = penalty_direct(image_first-image_second_warped)
    error += lambda_derivative * (penalty_spatial(derivative_x_func(flow_x))+penalty_spatial(derivative_y_func(flow_y)))
    
    neighboorhood = range(-math.floor(neighboor_size/2),math.ceil(neighboor_size/2))
    
    abs_error = np.zeros(shape=error.shape)
    
    for y in neighboorhood:
        for x in neighboorhood:
            shifted_flow_x = shift_array(flow_x,x,y)
            shifted_flow_y = shift_array(flow_y,x,y)
            abs_error += np.abs(flow_x-shifted_flow_x)
            abs_error += np.abs(flow_y-shifted_flow_y)
            
    return error + lambda_flow * abs_error

def penalty_squard(x):
    return x*x

def penalty_charbonnier(x):
    epsilon = 0.001
    a = 0.45
    return (x*x + epsilon*epsilon)**a



