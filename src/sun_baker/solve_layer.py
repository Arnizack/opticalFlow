import numpy as np
from src.sun_baker.solver_settings import SolverSettings
from src.utilities.color2grayscale import color2grayscale
from src.sun_baker.derivative_sun import derivative_sun
from src.sun_baker.setup_linear_system.setup_linear_system import setup_linear_system
from src.utilities.penalty_functions.IPenalty import IPenalty
from src.filter.cython.bilateral_median import bilateral_median_filter
from src.utilities.compute_occlusion import compute_occlusion
from src.utilities.warp_grid import warp_image
import scipy.sparse.linalg as splinalg
import math
from src.utilities.flow_field_helper import show_flow_field
import matplotlib.pyplot as plt
from time import time
from scipy.signal import medfilt2d



def solve_layer(first_image : np.ndarray, second_image : np.ndarray, initial_flow_field : np.ndarray,
                penalty_func : IPenalty, settings : SolverSettings ):
    """

    :param first_image: (ColorChannel, Height, Width)
    :param second_image:  (ColorChannel, Height, Width)
    :param settings: SolverSettings
    :return: Flowfield
    """

    width = first_image.shape[2]
    height = first_image.shape[1]

    second_image_warped = warp_image(second_image, initial_flow_field)

    first_gray_image = color2grayscale(first_image)
    second_gray_image_warped = color2grayscale(second_image_warped)

    I_x, I_y, I_t =  derivative_sun(first_gray_image,second_gray_image_warped)

    """
    plt.imshow(I_x)
    plt.figure()
    plt.imshow(I_y)
    plt.figure()
    plt.imshow(I_t)

    plt.show()
    """
    I_x.shape = (height * width)
    I_y.shape = (height * width)
    I_t.shape = (height * width)





    relax_flow_field = np.zeros(shape=(width*height*2),dtype=np.float32)

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

        x,info=splinalg.cg(A,b,tol = 0.001,maxiter=settings.maxiter_solve ,x0=guess_vu)[:2]



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



        #x,info=splinalg.lsmr(A,b)[:2]


        guess_vu = x
        flow = initial_flow_field+x.reshape(2,height,width)

        #plt.title("Before Filter")
        #show_flow_field(flow, width, height)
        #plt.show()

        init_flow = np.zeros(shape=(2, height, width), dtype=np.float32)

        occlusion = compute_occlusion(first_image, first_image, flow)
        #aux = np.zeros(shape=(2,height,width ))
        flow = bilateral_median_filter(flow.astype(np.float32), occlusion.astype(np.float32),
                                       flow.astype(np.float32), first_image.astype(np.float32),
                                       weigth_auxiliary=lambda_relax, weigth_filter=1,
                                       sigma_distance=settings.flow_filter_sigma_distance,
                                       sigma_color=settings.flow_filter_sigma_color,
                                       filter_size = settings.flow_filter_filter_size)


        #flow[0] = medfilt2d(flow[0], settings.median_filter_size)
        #flow[1] = medfilt2d(flow[1],settings.median_filter_size)

        #plt.title("After Filter")
        #show_flow_field(flow, width, height)
        #plt.show()

        relax_flow_field = flow.reshape(width*height*2)-initial_flow_field.reshape(width*height*2)
        guess_vu = relax_flow_field
        #relax_flow_field = x.reshape(width*height*2)


    return flow.reshape(2,height,width)
