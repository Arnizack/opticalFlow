from scipy.interpolate import interp2d
import numpy as np
from src.filter.gaussian_blur import gaussian_blur_matrix,gaussian_blur_flow
from scipy import ndimage
from src.utilities.flow_field_helper import show_flow_field
import matplotlib.pyplot as plt

def upscale_flow(flow,target_width,target_heigth):
    """

    :param flow: np.array(float) (YX,Height, Width)
    :param target_width: int >= Width
    :param target_heigth: int >= Height
    :return: np.array(float) (YX,Height,Width)
    """
    width = flow.shape[2]
    height = flow.shape[1]

    factor = target_width/width

    x_coords = np.arange(width)
    y_coords = np.arange(height)

    #grid_X,grid_Y = np.meshgrid(np.arange(width),np.arange(height))
    interpolate_func_Y = interp2d(x_coords,y_coords,flow[0], kind='cubic')
    interpolate_func_X = interp2d(x_coords, y_coords, flow[1], kind='cubic')

    width_upscale = np.linspace(0,width,target_width)
    height_upscale = np.linspace(0,height,target_heigth)

    upscale_flow_Y = interpolate_func_Y(width_upscale,height_upscale)
    upscale_flow_X = interpolate_func_X(width_upscale,height_upscale)

    return np.array([upscale_flow_Y*factor,upscale_flow_X*factor])



def down_scale_flow(flow,width,height):

    scale_x = width/flow.shape[2]
    scale_y = height/flow.shape[1]
    factor = (scale_x+scale_y)/2
    sigma = 1 / np.sqrt(2 * factor)

    flow_blur_y,flow_blur_x = gaussian_blur_flow(flow,sigma)

    scaled_flow_Y = ndimage.zoom(flow_blur_y, factor, order=0)
    scaled_flow_X = ndimage.zoom(flow_blur_x, factor, order=0)

    return np.array([scaled_flow_Y,scaled_flow_X])*factor