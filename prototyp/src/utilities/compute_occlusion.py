from src.utilities.image_derivative import differentiate_matrix
from src.utilities.warp_grid import warp_image
import numpy as np
import math

def compute_occlusion_log(first_frame,second_frame, flow, sigma_d = 0.3, sigma_e = 20):
    """

    :param first_frame: np.array(float) (ColorChannel, Height, Width)
    :param second_frame: np.array(float) (ColorChannel, Height, Width)
    :param flow: np.array(float) (YX, Height, Width)
    :param sigma_d: float
    :param sigma_e: float
    :return: occlusion map
    """
    #See EQ. 11. A Quantitative Analysis of Current Practices in Optical Flow Estimation
    #and the Principles behind Them

    flow_y_derivative = differentiate_matrix(flow[0])

    flow_x_derivative = differentiate_matrix(flow[1])


    div = flow_y_derivative[0] + flow_x_derivative[1]

    width = flow.shape[2]
    height = flow.shape[1]

    #d = min(div,0)
    d = div
    d[d>=0]=0


    X,Y = np.meshgrid(np.arange(width),np.arange(height))

    exponent = d**2/(2*sigma_d**2)

    first_frame_warped = warp_image(first_frame,flow)

    color_difference = first_frame - first_frame_warped

    exponent += np.linalg.norm(color_difference,axis=0)**2 / (2*sigma_e**2)

    return -exponent


def compute_occlusion(first_frame,second_frame, flow, sigma_d = 0.3, sigma_e = 20):
    """

    :param first_frame: np.array(float) (ColorChannel, Height, Width)
    :param second_frame: np.array(float) (ColorChannel, Height, Width)
    :param flow: np.array(float) (YX, Height, Width)
    :param sigma_d: float
    :param sigma_e: float
    :return: occlusion map
    """
    exponent = compute_occlusion_log(first_frame,second_frame,flow,sigma_d,sigma_e)
    occlusion = np.exp(exponent)

    return occlusion