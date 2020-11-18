import numpy as np

def normalize_flow(flow):
    """

    :param flow: (YX,Height, Width)
    :return: (YX,Height, Width)
    """
    Y,X = flow

    length = np.sqrt(Y**2+X**2)
    return np.array([Y/length,X/length])


def angular_error(flow1,flow2):
    """

    :param flow1: (YX,Height,Width)
    :param flow2: (YX,Height,Width)
    :return: (Height,Width)
    """
    flow1=normalize_flow(flow1)
    flow2 = normalize_flow(flow2)

    #dot product
    flow_dot = flow1*flow2
    flow_dot = flow_dot[0]+flow_dot[1]

    #get angle
    angular_error_flow = np.arccos(flow_dot)
    return angular_error_flow
