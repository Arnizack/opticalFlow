import numpy as np
import matplotlib.pyplot as plt
from src.utilities.image_pyramid import downscale_image
from src.utilities.warp_grid import warp_image
from src.utilities.image_access import show_image
from src.utilities.flow_field_helper import show_flow_field,show_flow_field_arrow
from src.utilities.absolute_endpoint_error import absolute_endpoint_error
from src.utilities.angular_error import angular_error
import scipy.ndimage as ndimage
from scipy import signal
from math import sqrt

def scale_flow(flow,width,height):
    scale_x = width/flow.shape[2]
    scale_y = height/flow.shape[1]
    factor = (scale_x+scale_y)/2
    sigma = 1 / sqrt(2 * factor)
    #flow_blur_y = ndimage.gaussian_filter(flow[0],sigma)
    #flow_blur_x = ndimage.gaussian_filter(flow[1], sigma)
    scaled_flow_Y = ndimage.zoom(flow[0], factor, order=0)
    scaled_flow_X = ndimage.zoom(flow[1], factor, order=0)
    return np.array([scaled_flow_Y,scaled_flow_X])

def scale_image(image, width, height):
    scale_x = width / image.shape[2]
    scale_y = height / image.shape[1]
    scale = (scale_x+scale_y)/2
    return downscale_image(image, scale)

def compare_flow(computed_flow, real_flow,current_frame,next_frame, plot=True,arrows=True):
    """

    :param computed_flow: np.array(float) (YX, height_1,width_1)
    :param real_flow: np.array(float) (YX, height_2,width_2)
    :param plot: bool
    :param current_frame: (ChannelCount, height,width)
    :param next_frame: (ChannelCount, height,width)
    :return: (min(height_1,height_2),min(width_1,width_2))
    """
    height_1 = computed_flow.shape[1]
    width_1 = computed_flow.shape[2]
    height_2 = real_flow.shape[1]
    width_2 = real_flow.shape[2]

    width = -1
    height = -1

    if(height_1>height_2 and width_1>width_2):
        computed_flow = scale_flow(computed_flow, width_2, height_2)
        width = width_2
        height = height_2
    elif (height_2>height_1 and width_2>width_1):
        real_flow = scale_flow(real_flow, width_1, height_1)
        width = width_1
        height = height_1

    elif(height_2==height_1 and width_2==width_1):
        width = width_1
        height=height_1
    else:
        raise Exception("False dimensions: flow_1: (", height_1, ", ", width_1, ") flow_2: (", height_2, ", ", width_2, ")" )

    ang_error = angular_error(computed_flow, real_flow)
    abs_error = absolute_endpoint_error(computed_flow, real_flow)

    print("Average angular error:")
    print(np.mean(ang_error[~np.isnan(ang_error)]))
    print("Absolute endpoint error:")
    print(np.mean(abs_error[~np.isnan(abs_error)]))

    if(plot):
        #rescale image
        current_frame = scale_image(current_frame,width,height)
        next_frame = scale_image(next_frame, width, height)

        next_frame_warped = warp_image(next_frame,computed_flow)


        plt.figure()
        show_image(current_frame)
        plt.title("Current frame")

        plt.figure()
        show_image(next_frame_warped)
        plt.title("Next frame warped to current frame")

        fig, axs = plt.subplots(1,2)
        axs[0].set_title("Current frame - Next frame squared")
        show_image((current_frame-next_frame)**2,axes=axs[0])

        axs[1].set_title("Current frame - Next frame warped squared")
        show_image((current_frame - next_frame_warped) ** 2,axes=axs[1])

        if (arrows):
            plt.figure()
            show_image(current_frame, axes=plt)
            show_flow_field_arrow(computed_flow, width, height, axes=plt)
            plt.title("Arrows")

        fig, axs = plt.subplots(1,2)

        show_flow_field(computed_flow,width,height,axes=axs[0])
        axs[0].set_title("Computed_flow")
        show_flow_field(real_flow, width, height, axes=axs[1])
        axs[1].set_title("Ground Truth")




        fig, axs = plt.subplots(1,2)
        axs[0].imshow(ang_error)
        axs[0].set_title("Angular error")
        axs[1].imshow(abs_error)
        axs[1].set_title("Absolute endpoint error")

        plt.show()


