
import numpy as np
cimport numpy as np
from libc.math cimport exp, floor,ceil
from cython.parallel import prange

from cpython.array cimport array
from libc.stdlib cimport qsort
from libc.stdlib cimport malloc, free

cdef int cmp_func(const void* a, const void* b) nogil:
    cdef float a_v = (<float*>a)[0]
    cdef float b_v = (<float*>b)[0]
    if a_v < b_v:
        return -1
    elif a_v == b_v:
        return 0
    else:
        return 1

cdef void sort_c(float* a, int length) nogil:
    # a needn't be C continuous because strides helps
    qsort(a, length, sizeof(float), &cmp_func)

cdef float quickselect_median(float* l, int length) nogil:
    cdef int half = ( length -1 ) // 2
    sort_c(l,length)
    if(length % 2 == 0):
        return l[half + 1] * 0.5 + l[half] *0.5
    return l[half]

cdef int min(int a, int b) nogil:
    if(a > b):
        return b
    return a

cdef int max(int a, int b) nogil:
    if(a > b):
        return a
    return b

cpdef np.ndarray[np.float32_t, ndim=3] bilateral_median_filter(float[:,:, ::1] flow, float[:, ::1] occlusen, 
                            float[:,:, ::1] auxiliary_field, float[:,:, ::1] image, 
                            float weigth_auxiliary, float weigth_filter,
                            float sigma_distance = 7, float sigma_color = float(7/200), int filter_size=5):
    """

    :param flow: np.float (YX,Height,Width)
    :param occlusen: (Height, Width)
    :param auxiliary_field: np.array(float) (Y_flow X_flow , Height, Width)
    :param image: np.array(float) (ColorChannel, Height, Width)
    :param weigth_auxiliary: float > 0
    :param weigth_filter: float > 0
    :param sigma_distance: float
    :param sigma_color: float
    :param filter_size: int
    :return: flow field
    """
    print("Cython Bilateral median filter")
    cdef int width = flow.shape[2]
    cdef int height = flow.shape[1]
    cdef int color_channel_count = image.shape[0]

    cdef int filter_half = int(filter_size / 2)

    cdef int helper_list_size = filter_size * filter_size * 2
 

    cpdef np.ndarray[np.float32_t, ndim=3] result_flow = np.empty(shape=(2, height, width), dtype=np.float32)

    cdef float* helper_flow_x_list
    cdef float* helper_flow_y_list
    cdef float* weigths_list 
    cdef int min_x_compare
    cdef int max_x_compare
    cdef int min_y_compare
    cdef int max_y_compare
    cdef int counter
    cdef float distance_squared_difference
    cdef float color_squared_difference
    cdef float exponent
    cdef float occlusen_current
    cdef float occlusen_compared
    cdef float weigth
    cdef float f_x
    cdef float f_y
    cdef float scalar
    cdef float sum
    cdef int channel_idx
    cdef int n,x,y,x_compare,y_compare,idx_1,idx_2

    #for y in prange(height, nogil=True):
    for y in range(height):
        helper_flow_x_list =  <float *> malloc(helper_list_size * sizeof(float))
        helper_flow_y_list = <float *> malloc(helper_list_size * sizeof(float))
        weigths_list = <float *> malloc(helper_list_size * sizeof(float))
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
                    for channel_idx in range(color_channel_count):
                        color_squared_difference += (image[channel_idx,y_compare,x_compare] - image[channel_idx,y,x]) ** 2

                    exponent = distance_squared_difference / 2 * sigma_distance
                    exponent += color_squared_difference / 2 * sigma_color * color_channel_count

                    occlusen_current = occlusen[y,x]
                    occlusen_compared = occlusen[y_compare,x_compare]

                    weigth = exp(-exponent) * occlusen_compared / occlusen_current
                    weigths_list[counter] = weigth

                    helper_flow_x_list[counter] = flow[1,y_compare,x_compare]
                    helper_flow_y_list[counter] = flow[0,y_compare,x_compare]

                    counter += 1

            # See A NEW MEDIAN FORMULA WITH APPLICATIONS TO PDE BASED DENOISING
            # 3.13

            n = counter

            f_x = auxiliary_field[1,y,x]
            f_y = auxiliary_field[0,y,x]
            scalar = weigth_filter / weigth_auxiliary

            for idx_1 in range(n):
                sum = 0
                for idx_2 in range(idx_1):
                    sum -= weigths_list[idx_2]

                for idx_2 in range(idx_1, n):
                    sum += weigths_list[idx_2]
                helper_flow_x_list[n + idx_1] = f_x + scalar * sum
                helper_flow_y_list[n + idx_1] = f_y + scalar * sum

            result_flow[0,y,x] = quickselect_median(helper_flow_y_list,n*2)
            result_flow[1,y,x] = quickselect_median(helper_flow_x_list,n*2)
        free(helper_flow_x_list)
        free(helper_flow_y_list)
        free(weigths_list)

    return result_flow
