import numpy as np

def absolute_endpoint_error(flow1,flow2):
    """

    :param flow1: (YX,Height,Width)
    :param flow2: (YX,Height,Width)
    :return: (Height,Width)
    """

    Y1,X1 = flow1
    Y2,X2 = flow2

    distance_error = np.sqrt((Y1-Y2)**2+(X1-X2)**2)
    return distance_error