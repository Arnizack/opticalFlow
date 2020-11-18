
import numpy as np
cimport numpy as np
from libc.math cimport exp, floor,ceil
from cython.parallel import prange

from cpython.array cimport array
from libc.stdlib cimport qsort
from libc.stdlib cimport malloc, free
cimport cython


cdef int cmp_func(const void* a, const void* b) nogil:
    cdef double a_v = (<double*>a)[0]
    cdef double b_v = (<double*>b)[0]
    if a_v < b_v:
        return -1
    elif a_v == b_v:
        return 0
    else:
        return 1

cdef void sort_c(double* a, int length) nogil:
    # a needn't be C continuous because strides helps
    qsort(a, length, sizeof(double), &cmp_func)

cdef double quickselect_median(double* l, int length) nogil:
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

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cdef void inner_row_kernel(int y, double[:,:, ::1] flow, double[:, ::1] log_occlusen, 
double[:,:, ::1] auxiliary_field, 
double[:,:, ::1] image,double[:,:,::1] result_flow,
                        double weigth_auxiliary, double weigth_filter, int width, int height,
                        double sigma_distance, double sigma_color, int filter_size, int color_channel_count, int filter_half, int helper_list_size
                ) nogil:
    cdef double* helper_flow_x_list
    cdef double* helper_flow_y_list
    cdef double* weigths_list 
    cdef int min_x_compare
    cdef int max_x_compare
    cdef int min_y_compare
    cdef int max_y_compare
    cdef int counter
    cdef double distance_squared_difference
    cdef double color_squared_difference
    cdef double exponent
    cdef double occlusen_current
    cdef double occlusen_compared
    cdef double weigth
    cdef double f_x
    cdef double f_y
    cdef double scalar
    cdef double sum
    cdef int channel_idx
    cdef int n,x,x_compare,y_compare,idx_1,idx_2

    helper_flow_x_list =  <double *> malloc((helper_list_size+1) * sizeof(double))
    helper_flow_y_list = <double *> malloc((helper_list_size+1) * sizeof(double))
    weigths_list = <double *> malloc(helper_list_size * sizeof(double))
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

                exponent = distance_squared_difference / (2 * sigma_distance * sigma_distance)
                exponent += color_squared_difference / (2 * sigma_color * sigma_color * color_channel_count)

                occlusen_current = log_occlusen[y,x]
                occlusen_compared = log_occlusen[y_compare,x_compare]

                weigth = exp(-exponent+occlusen_compared-occlusen_current)
                weigths_list[counter] = weigth

                helper_flow_x_list[counter] = flow[1,y_compare,x_compare]
                helper_flow_y_list[counter] = flow[0,y_compare,x_compare]

                counter += 1

        # See A NEW MEDIAN FORMULA WITH APPLICATIONS TO PDE BASED DENOISING
        # 3.13

        n = counter

        f_x = auxiliary_field[1,y,x]
        f_y = auxiliary_field[0,y,x]
        scalar = 1/(2*(weigth_auxiliary / weigth_filter))

        for idx_1 in range(n+1):
            sum = 0
            for idx_2 in range(idx_1):
                sum -= weigths_list[idx_2]

            for idx_2 in range(idx_1, n):
                sum += weigths_list[idx_2]
            helper_flow_x_list[n + idx_1] = f_x + scalar * sum
            helper_flow_y_list[n + idx_1] = f_y + scalar * sum

        result_flow[0,y,x] = quickselect_median(helper_flow_y_list,n*2+1)
        result_flow[1,y,x] = quickselect_median(helper_flow_x_list,n*2+1)
    free(helper_flow_x_list)
    free(helper_flow_y_list)
    free(weigths_list)

cpdef np.ndarray[np.double_t, ndim=3] bilateral_median_filter(double[:,:, ::1] flow, double[:, ::1] log_occlusen, 
                            double[:,:, ::1] auxiliary_field, double[:,:, ::1] image, 
                            double weigth_auxiliary, double weigth_filter,
                            double sigma_distance = 7, double sigma_color = <double> 7.0/200, int filter_size=5):
    """

    :param flow: np.double (YX,Height,Width)
    :param log_occlusen: (Height, Width)
    :param auxiliary_field: np.array(double) (Y_flow X_flow , Height, Width)
    :param image: np.array(double) (ColorChannel, Height, Width)
    :param weigth_auxiliary: double > 0
    :param weigth_filter: double > 0
    :param sigma_distance: double
    :param sigma_color: double
    :param filter_size: int
    :return: flow field
    """
    print("Cython Bilateral median filter")
    
    cdef int width = flow.shape[2]
    cdef int height = flow.shape[1]
    cdef int color_channel_count = image.shape[0]

    cdef int filter_half = int(filter_size / 2)

    cdef int helper_list_size = filter_size * filter_size * 2
 

    cpdef np.ndarray[np.double_t, ndim=3] result_flow = np.empty(shape=(2, height, width), dtype=np.double)
    print("Channelcount: ", color_channel_count, " Sigma Color: ", sigma_color, " Sigma distance: ", sigma_distance)
    cdef int y
    #for y in prange(height, nogil=True):
    for y in range(height):
        inner_row_kernel(y,flow, log_occlusen, auxiliary_field, image,result_flow,
                        weigth_auxiliary, weigth_filter,width,height,
                        sigma_distance, sigma_color , filter_size, color_channel_count,filter_half, helper_list_size)

    return result_flow
