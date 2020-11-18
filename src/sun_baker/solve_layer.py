import numpy as np
from src.sun_baker.solver_settings import SolverSettings
from src.utilities.color2grayscale import color2grayscale
from src.sun_baker.derivative_sun import derivative_sun
from src.sun_baker.setup_linear_system.setup_linear_system import setup_linear_system
from src.utilities.penalty_functions.IPenalty import IPenalty
#from src.filter.cython.bilateral_median import bilateral_median_filter
from src.filter.bilater_median import bilateral_median_filter
from src.utilities.compute_occlusion import compute_occlusion,compute_occlusion_log
from src.utilities.warp_grid import warp_image, warp_matrix
import scipy.sparse.linalg as splinalg
import math
from src.utilities.flow_field_helper import show_flow_field
import matplotlib.pyplot as plt
from time import time
from scipy.signal import medfilt2d



def solve_layer(filter_image : np.ndarray,first_gray_image : np.ndarray, second_gray_image : np.ndarray,
                initial_flow_field : np.ndarray,
                penalty_func : IPenalty, settings : SolverSettings ):
    """

    :param filter_image: (ColorChannel, Height, Width)
    :param first_gray_image: (Height, Width)
    :param second_gray_image: (Height, Width)
    :param initial_flow_field: (YX,Height,Width)
    :param penalty_func: IPenalty
    :param settings: SolverSettings
    :return: Flowfield
    """

    width = filter_image.shape[2]
    height = filter_image.shape[1]

    second_gray_image_warped = warp_matrix(second_gray_image, initial_flow_field)

    I_x, I_y, I_t =  derivative_sun(first_gray_image,second_gray_image_warped)


    #plt.imshow(first_gray_image)
    #plt.figure()
    #plt.imshow(second_gray_image)
    #plt.figure()
    #plt.imshow(second_gray_image_warped)
    #plt.figure()
    #plt.imshow(I_x)
    #plt.figure()
    #plt.imshow(I_y)
    #plt.figure()
    #plt.imshow(I_t)
    #plt.figure()
    #plt.imshow(second_gray_image_warped-first_gray_image)

    #plt.show()

    I_x.shape = (height * width)
    I_y.shape = (height * width)
    I_t.shape = (height * width)

    relax_flow_field = np.zeros(shape=(width*height*2),dtype=np.double)

    relaxation_steps = settings.relaxation_steps

    lambda_k = settings.weight_kernel

    start_relax = settings.weight_relaxation_start
    end_relax = settings.weight_relaxation_end

    if(relaxation_steps> 1):
        lambda_relax_func = lambda x : math.log(math.exp(start_relax)+(math.exp(end_relax)-math.exp(start_relax))/(relaxation_steps-1) * x)

    else:
        lambda_relax_func = lambda x: start_relax

    guess_vu = np.zeros(shape=(height*width*2))

    last_x = np.zeros(shape=(height*width*2))

    for relaxation_iter in range(relaxation_steps):
        lambda_relax = lambda_relax_func(relaxation_iter)

        print("Lambda relax: ",lambda_relax)

        A,b = setup_linear_system(I_x, I_y, I_t, guess_vu, relax_flow_field, settings.kernel, width,height,
                            lambda_relax, lambda_k**2, penalty_func)

        start = time()

        x,info= splinalg.cg(A, b, tol = 0.001, maxiter=settings.max_iter_solve, x0=guess_vu)[:2]



        print("Lg with cg: ", time() - start)
        print("Number of iterations: ", info)
        print("Mean error: ", (b - A.dot(x)).mean())

        norm = np.linalg.norm(x - last_x)
        last_x = x
        print("Norm: ",norm)
        if (norm < 1e-03):
            print("Terminate iteration early\n\n")
            flow = x
            break


        guess_vu = x
        relax_flow_field = x
        flow = initial_flow_field+x.reshape(2,height,width)

        #plt.title("Before Filter")
        #show_flow_field(flow, width, height)
        #plt.show()

        if(settings.flow_filter_filter_size>3):
            print("Flow")
            print(flow)
            print("FilterImage")
            print(filter_image)
            init_flow = np.zeros(shape=(2, height, width), dtype=np.double)

            log_occlusion = compute_occlusion_log(filter_image, filter_image, flow)

            print("Log Occlusion")
            print(log_occlusion)

            flow = bilateral_median_filter(flow.astype(np.double), log_occlusion.astype(np.double),
                                           flow.astype(np.double), filter_image.astype(np.double),
                                           weigth_auxiliary=lambda_relax, weigth_filter=5,
                                           sigma_distance=settings.flow_filter_sigma_distance,
                                           sigma_color=settings.flow_filter_sigma_color,
                                           filter_size = settings.flow_filter_filter_size)

            #plt.title("After Filter")
            #show_flow_field(flow, width, height)
            #plt.show()

            print("flow after")
            print(flow)

            relax_flow_field = flow.reshape(width*height*2)-initial_flow_field.reshape(width*height*2)
            guess_vu = relax_flow_field


    return flow.reshape(2,height,width)
