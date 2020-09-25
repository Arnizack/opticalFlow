import math
import numpy as np
from statistics import median
from src.filter.median import quickselect_median


def bilateral_median_filter(flow, log_occlusen, auxiliary_field, image, weigth_auxiliary, weigth_filter,
                            sigma_distance = 7, sigma_color =7 / 200, filter_size=5):
    """

    :param flow: np.float (YX,Height,Width)
    :param occlusen: (Height, Width)
    :param auxiliary_field: np.array(float) (Y_flow X_flow , Y_coord X_coord, Height, Width)
    :param image: np.array(float) (ColorChannel, Height, Width)
    :param weigth_auxiliary: float > 0
    :param weigth_filter: float > 0
    :param sigma_distance: float
    :param sigma_color: float
    :param filter_size: int
    :return: flow field
    """
    width = flow.shape[2]
    height = flow.shape[1]
    color_channel_count = flow.shape[0]

    filter_half = int(filter_size / 2)

    helper_list_size = filter_size ** 2 * 2
    helper_flow_x_list = [0.0] * (helper_list_size+1)
    helper_flow_y_list = [0.0] * (helper_list_size+1)
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
                    exponent += color_squared_difference / 2 * sigma_color * color_channel_count

                    occlusen_current = log_occlusen[y][x]
                    occlusen_compared = log_occlusen[y_compare][x_compare]

                    #weigth = math.exp(-exponent) * occlusen_compared / occlusen_current
                    weigth = math.exp(-exponent+occlusen_compared-occlusen_current)
                    weigths_list[counter] = weigth

                    helper_flow_x_list[counter] = flow[1][y_compare][x_compare]
                    helper_flow_y_list[counter] = flow[0][y_compare][x_compare]

                    counter += 1

            # See A NEW MEDIAN FORMULA WITH APPLICATIONS TO PDE BASED DENOISING
            # 3.13

            n = counter

            f_x = auxiliary_field[1][y][x]
            f_y = auxiliary_field[0][y][x]
            scalar = 1/(2*(weigth_auxiliary / weigth_filter))

            for idx_1 in range(n+1):
                sum = 0
                for idx_2 in range(idx_1):
                    sum -= weigths_list[idx_2]

                for idx_2 in range(idx_1, n):
                    sum += weigths_list[idx_2]
                helper_flow_x_list[n + idx_1] = f_x + scalar * sum
                helper_flow_y_list[n + idx_1] = f_y + scalar * sum

            result_flow[0][y][x] = median(helper_flow_y_list[:n*2+1])
            result_flow[1][y][x] = median(helper_flow_x_list[:n*2+1])

    return result_flow
