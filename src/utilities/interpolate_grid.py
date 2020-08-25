from  scipy import interpolate

import numpy as np

def interpolate_grid(data_grid,offset):
    """

    :param data_grid: np.array(float) (Width, Height)
    :param offset: np.array(float) (2, Width, Height)
    :return: interpolate obj
    """
    xdim = data_grid.shape[0]
    ydim = data_grid.shape[1]

    x_coords, y_coords = np.meshgrid(np.arange(xdim),np.arange(ydim))
    x_coords-= offset[0]
    y_coords-= offset[1]

    return interpolate.interp2d(x_coords,y_coords,data_grid)
