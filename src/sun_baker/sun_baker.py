import numpy as np
import matplotlib.pyplot as plt
from src.utilities.flow_field_helper import show_flow_field
from src.sun_baker.solver_settings import SolverSettings
from src.utilities.image_pyramid import create_image_pyramid
from src.utilities.scale_flow_field import upscale_flow,down_scale_flow
from src.sun_baker.solve_layer import solve_layer
from time import time
from src.utilities.penalty_functions.SquaredPenalty import SquaredPenalty
from src.utilities.penalty_functions.MixPenalty import MixPenalty
from src.utilities.penalty_functions.GeneralizedCharbonnierPenalty import GeneralizedCharbonnierPenalty
from src.utilities.color2grayscale import color2grayscale
from src.preprocessing.rof import denoising_chambolle
from scipy.signal import medfilt2d


def sun_baker_optical_flow(first_image : np.ndarray, second_image : np.ndarray,settings = SolverSettings() ) -> np.ndarray:
    start_time = time()

    factors = settings.scale_factors
    pyramid_levels_first_image = create_image_pyramid(first_image, factors)
    pyramid_levels_second_image = create_image_pyramid(second_image, factors)

    width = pyramid_levels_first_image[0].shape[2]
    height = pyramid_levels_first_image[0].shape[1]

    gnc_steps = settings.gnc_steps

    flow = np.zeros(shape=(2, height, width))

    penalty_func_1 = SquaredPenalty()
    penalty_func_2 = GeneralizedCharbonnierPenalty()

    penalty_func = MixPenalty(penalty_func_1, penalty_func_2, 0)

    relaxation_steps = settings.relaxation_steps
    maxiter_solve = settings.maxiter_solve

    for first_frame_scaled, second_frame_scaled in zip(pyramid_levels_first_image, pyramid_levels_second_image):
        width = first_frame_scaled.shape[2]
        height = first_frame_scaled.shape[1]
        flow = upscale_flow(flow, width, height)
        first_gray_frame_scaled = color2grayscale(first_frame_scaled)
        second_gray_frame_scaled = color2grayscale(second_frame_scaled)

        for gnc_iter in range(gnc_steps):
            #width = pyramid_levels_first_image[0].shape[2]
            #height = pyramid_levels_first_image[0].shape[1]


            print("-------- GNC: ", gnc_iter," --------")

            if (gnc_steps > 1):
                mix_factor = gnc_iter / (gnc_steps - 1)
            else:
                mix_factor = 0

            penalty_func.mix_factor = mix_factor



            if(mix_factor==0):
                settings.relaxation_steps = 1
                settings.maxiter_solve = 100
            else:
                settings.relaxation_steps = relaxation_steps
                settings.maxiter_solve = maxiter_solve

            #if(flow.shape[2] != width or flow.shape[1] != height):
            #    flow = down_scale_flow(flow,width,height)

            for iter in range(settings.steps_per_level):
                flow = solve_layer(first_frame_scaled,second_frame_scaled,first_gray_frame_scaled,second_gray_frame_scaled,flow, penalty_func,settings)

            #flow[0] = medfilt2d(flow[0], settings.median_filter_size)
            #flow[1] = medfilt2d(flow[1],settings.median_filter_size)
        #plt.title("At level: "+str(width) +","+str(height))
        #show_flow_field(flow, width, height)
        #plt.show()

    print("Sun Baker full time: ", time() - start_time)
    return flow