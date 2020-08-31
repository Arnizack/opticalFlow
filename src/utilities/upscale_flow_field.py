from scipy.interpolate import interp2d
import numpy as np

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
