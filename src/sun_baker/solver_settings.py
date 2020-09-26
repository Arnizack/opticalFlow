import numpy as np

class SolverSettings:
    weight_kernel = 50/255
    weight_relaxation_start = 1e-03 / 255
    weight_relaxation_end =  100/ 255

    median_filter_size = 5

    gnc_steps = 3
    relaxation_steps = 10
    steps_per_level = 3

    max_iter_solve=10

    scale_factors = [0.8]
    gnc_scale_factors = [0.8]

    flow_filter_sigma_distance = 7
    flow_filter_sigma_color = 7/255
    flow_filter_filter_size = 7

    kernel = np.array(
        [[1, 2, 1],
         [2, -12, 2],
         [1, 2, 1]]) / 12