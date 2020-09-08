
import numpy as np
cimport numpy as np
from libc.math cimport exp
from cpython.array cimport array


cpdef float quickselect_median(float[:] l):
    if len(l) % 2 == 1:
        return quickselect(l, len(l) / 2)
    else:
        return 0.5 * (quickselect(l, len(l) / 2 - 1) +
                      quickselect(l, len(l) / 2))


cdef float quickselect(float[:] l, size_t k):
    """
    Select the kth element in l (0 based)
    :param l: List of numerics
    :param k: Index
    :param pivot_fn: Function to choose a pivot, defaults to random.choice
    :return: The kth element of l
    """
    
    cdef size_t length = len(l)
    if length == 1:
        assert k == 0
        return l[0]
    half = int(length/2)
    pivot = l[half]

    cdef float [:] lows = [el for el in l if el < pivot]
    return 0.0
    cdef float [:] highs = [el for el in l if el > pivot]
    cdef float [:] pivots = [el for el in l if el == pivot]

    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        # We got lucky and guessed the median
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots))

def bilateral_median_filter(flow, occlusen, auxiliary_field, image, weigth_auxiliary, weigth_filter,
                            sigma_distance = 7, sigman_color = 7/200, filter_size=5):
    """

    :param flow: np.float (YX,Height,Width)
    :param occlusen: (Height, Width)
    :param auxiliary_field: np.array(float) (Y_flow X_flow , Y_coord X_coord, Height, Width)
    :param image: np.array(float) (ColorChannel, Height, Width)
    :param weigth_auxiliary: float > 0
    :param weigth_filter: float > 0
    :param sigma_distance: float
    :param sigman_color: float
    :param filter_size: int
    :return: flow field
    """
    print("Cython Bilateral median filter")
    width = flow.shape[2]
    height = flow.shape[1]
    color_channel_count = flow.shape[0]

    filter_half = int(filter_size / 2)

    helper_list_size = filter_size ** 2 * 2
    
    c_helper_flow_x_list = np.empty(shape=(helper_list_size),dtype = np.float32)
    c_helper_flow_y_list = np.empty(shape=(helper_list_size),dtype = np.float32)

    cdef float [:] helper_flow_x_list = c_helper_flow_x_list
    cdef float [:] helper_flow_y_list = c_helper_flow_y_list
    weigths_list = [0.0] * helper_list_size

    result_flow = np.empty(shape=(2, height, width), dtype=float)

    for y in range(height):
        for x in range(width):
            min_x_compare = max(0, x - filter_half)
            max_x_compare = min(width, x + filter_half + 1)

            min_y_compare = max(0, y - filter_half)
            max_y_compare = min(height, y + filter_half + 1)

            counter = 0

            for y_compare in range(min_y_compare, max_y_compare):
                for x_compare in range(min_x_compare, max_x_compare):
                    distance_squared_difference = (y - y_compare) ** 2 + (x - x_compare) ** 2
                    color_squared_difference = 0
                    for channel in image:
                        color_squared_difference += (channel[y_compare][x_compare] - channel[y][x]) ** 2

                    exponent = distance_squared_difference / 2 * sigma_distance
                    exponent += color_squared_difference / 2 * sigman_color * color_channel_count

                    occlusen_current = occlusen[y][x]
                    occlusen_compared = occlusen[y_compare][x_compare]

                    weigth = exp(-exponent) * occlusen_compared / occlusen_current
                    weigths_list[counter] = weigth

                    helper_flow_x_list[counter] = flow[1][y_compare][x_compare]
                    helper_flow_y_list[counter] = flow[0][y_compare][x_compare]

                    counter += 1

            # See A NEW MEDIAN FORMULA WITH APPLICATIONS TO PDE BASED DENOISING
            # 3.13

            n = counter

            f_x = auxiliary_field[1][y][x]
            f_y = auxiliary_field[0][y][x]
            scalar = weigth_filter / weigth_auxiliary

            for idx_1 in range(n):
                sum = 0
                for idx_2 in range(idx_1):
                    sum -= weigths_list[idx_2]

                for idx_2 in range(idx_1, n):
                    sum += weigths_list[idx_2]
                helper_flow_x_list[n + idx_1] = f_x + scalar * sum
                helper_flow_y_list[n + idx_1] = f_y + scalar * sum

            result_flow[0][y][x] = quickselect_median(helper_flow_y_list[:n*2])
            result_flow[1][y][x] = quickselect_median(helper_flow_x_list[:n*2])

    return result_flow
