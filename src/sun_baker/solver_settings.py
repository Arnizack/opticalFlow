import numpy as np

class SolverSettings:
    weight_kernel = 40/200
    weight_relaxation_start = 1e-03 / 255
    weight_relaxation_end = 100 / 255
    #derivative = "Horn-Schunck"
    solver = "cg_own"
    derivative_mode = "gray"
    #"Horn-Schunck"
    derivative_typ = "Sun"
    median_filter_size = 7

    scale_factors = [0.5,0.5]

    kernel = np.array(
        [[1, 2, 1],
         [2, -12, 2],
         [1, 2, 1]]) / 12