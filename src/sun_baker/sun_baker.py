import numpy as np
import matplotlib.pyplot as plt
from src.utilities.flow_field_helper import show_flow_field
from src.sun_baker.solver_settings import SolverSettings
from src.utilities.image_pyramid import create_image_pyramid, create_matrix_pyramid
from src.utilities.scale_flow_field import upscale_flow,down_scale_flow
from src.sun_baker.solve_layer import solve_layer
from time import time
from src.utilities.penalty_functions.SquaredPenalty import SquaredPenalty
from src.utilities.penalty_functions.MixPenalty import MixPenalty
from src.utilities.penalty_functions.GeneralizedCharbonnierPenalty import GeneralizedCharbonnierPenalty
from src.utilities.color2grayscale import color2grayscale
from src.utilities.image_access import show_image
from src.preprocessing.rof import denoising_chambolle, denoising_chambolle_image
from src.utilities.scale_np_array_in_range import scale_image_channels_in_range,scale_np_array_in_range
from scipy.signal import medfilt2d
from typing import Tuple


def preprocessing(first_image : np.ndarray,second_image : np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """

    :param first_image: (Color,Height,Width)
    :param second_image: (Color,Height,Width)
    :return: (filter_image, first_gray_image, second_gray_image)
    """
    first_struc, first_tex = denoising_chambolle_image(first_image)
    second_struc, second_tex = denoising_chambolle_image(second_image)

    filter_image = scale_image_channels_in_range(first_struc,start=0,end=1)

    mix = 4/5

    first_denoise_img = first_tex*mix + first_struc*(1-mix)
    second_denoise_img = second_tex * mix + second_struc * (1 - mix)

    first_denoise_img = scale_image_channels_in_range(first_denoise_img,start=0,end=1)
    second_denoise_img = scale_image_channels_in_range(second_denoise_img,start=0,end=1)

    first_gray_image = color2grayscale(first_denoise_img)
    second_gray_image = color2grayscale(second_denoise_img)

    first_gray_image = scale_np_array_in_range(first_gray_image,start = 0,end = 1)
    second_gray_image = scale_np_array_in_range(second_gray_image,start = 0,end=1)

    return (filter_image,first_gray_image,second_gray_image)

def sun_baker_optical_flow(first_image : np.ndarray, second_image : np.ndarray,settings = SolverSettings() ) -> np.ndarray:
    start_time = time()

    factors = settings.scale_factors
    gnc_factors = settings.gnc_scale_factors

    filter_image, first_gray_image, second_gray_image = preprocessing(first_image,second_image)

    pyramid_levels_first_gray_image = create_matrix_pyramid(first_gray_image, factors)
    pyramid_levels_second_gray_image = create_matrix_pyramid(second_gray_image, factors)

    pyramid_levels_filter_image = create_image_pyramid(filter_image, factors)


    width = pyramid_levels_filter_image[0].shape[2]
    height = pyramid_levels_filter_image[0].shape[1]

    gnc_steps = settings.gnc_steps

    flow = np.zeros(shape=(2, height, width))

    penalty_func_1 = SquaredPenalty()
    penalty_func_2 = GeneralizedCharbonnierPenalty()

    penalty_func = MixPenalty(penalty_func_1, penalty_func_2, 0)

    relaxation_steps = settings.relaxation_steps
    max_iter_solve = settings.max_iter_solve

    for gnc_iter in range(gnc_steps):
        width = pyramid_levels_filter_image[0].shape[2]
        height = pyramid_levels_filter_image[0].shape[1]

        #downscale
        #if (flow.shape[2] > width or flow.shape[1] > height):


        if(gnc_iter>0):
            pyramid_levels_first_gray_image = create_matrix_pyramid(first_gray_image, gnc_factors)
            pyramid_levels_second_gray_image = create_matrix_pyramid(second_gray_image, gnc_factors)

            pyramid_levels_filter_image = create_image_pyramid(filter_image, gnc_factors)
            width = pyramid_levels_filter_image[0].shape[2]
            height = pyramid_levels_filter_image[0].shape[1]
            flow = down_scale_flow(flow, width, height)

        for filter_scaled,first_gray_scaled,second_gray_scaled in \
                zip(pyramid_levels_filter_image, pyramid_levels_first_gray_image, pyramid_levels_second_gray_image):

            width = filter_scaled.shape[2]
            height = filter_scaled.shape[1]
            flow = upscale_flow(flow, width, height)

            print("-------- GNC: ", gnc_iter," --------")

            if (gnc_steps > 1):
                mix_factor = gnc_iter / (gnc_steps - 1)


            else:
                mix_factor = 0

            penalty_func.mix_factor = mix_factor


            if(mix_factor==0):
                settings.relaxation_steps = 1
                settings.max_iter_solve = 100

            else:
                settings.relaxation_steps = relaxation_steps
                settings.max_iter_solve = max_iter_solve


            for iter in range(settings.steps_per_level):
                flow = solve_layer(filter_scaled,first_gray_scaled,second_gray_scaled,flow, penalty_func,settings)


            plt.title("At level: "+str(width) +","+str(height)+", GNC: "+str(gnc_iter))
            show_flow_field(flow, width, height)
            plt.show()

    print("Sun Baker full time: ", time() - start_time)
    return flow