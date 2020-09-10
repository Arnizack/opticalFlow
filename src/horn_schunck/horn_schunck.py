from src.utilities.image_pyramid import create_image_pyramid
from src.utilities.upscale_flow_field import upscale_flow
from src.horn_schunck.solve_layer import solve_layer, SolverSettings

import numpy as np
from time import time

def compute_optical_flow_HS(first_frame,second_frame,settings = SolverSettings()):
    start_time = time()

    factors = settings.scale_factors
    pyramid_levels_first_frame = create_image_pyramid(first_frame,factors)
    pyramid_levels_second_frame = create_image_pyramid(second_frame,factors)

    width = pyramid_levels_first_frame[0].shape[2]
    height = pyramid_levels_first_frame[0].shape[1]

    flow = np.zeros(shape=(2,height,width))

    for first_frame_scaled,second_frame_scaled in zip(pyramid_levels_first_frame,pyramid_levels_second_frame):
        width = first_frame_scaled.shape[2]
        height = first_frame_scaled.shape[1]
        flow = upscale_flow(flow,width,height)
        for iter in range(3):
            flow = solve_layer(first_frame_scaled,second_frame_scaled,flow,settings)



    print("Horn Schunck full time: ",time()-start_time)
    return flow
