import numpy as np

class SolverSettings:
    weight_kernel = 10/255
    weight_relaxation_start = 1e-04 / 255
    weight_relaxation_end =  1e-01/ 255

    median_filter_size = 5

    gnc_steps = 1
    relaxation_steps = 4
    steps_per_level = 3

    max_iter_solve=50

    scale_factors = [0.8,0.5,0.5,0.5,0.5,0.5,0.5]
    gnc_scale_factors = [0.8]

    flow_filter_sigma_distance = 7
    flow_filter_sigma_color = 7/255
    flow_filter_filter_size = 15

    kernel = np.array(
        [[1, 2, 1],
         [2, -12, 2],
         [1, 2, 1]]) / 12