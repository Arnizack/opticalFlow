import numpy as np
from scipy import signal
from time import time


def denoising_chambolle(img, lambda0=0.125, std_dev = None, iter=100, text_factor = 0.95):
    """
    denoises the Image and calculates the textured part (difference)
    reference: "An algorithm for total variation minimization and applications" (Antonin Chambolle) Chapter 4
    :param img: gray scale CHW image
    :param lambda0: lambda factor > 0
    :param std_dev: approximated std of the Image Noise
    :param iter: number of maximum Iterations
    :param text_factor: influence factor of the deionised image
    :return: denoised image (structured part), textured part
    """
    print("Denoising started")
    start = time()
    if std_dev == None:
        std_dev = np.std(img)
    size = img.size
    perfect_solution = np.sqrt(size * std_dev)
    max_solution = np.linalg.norm(img - img.mean())

    if perfect_solution <= 0:
        perfect_solution = 1
        print("perfect solution is less then 0")

    if perfect_solution >= max_solution:
        perfect_solution = max_solution
        print("perfect solution is to big", max_solution, ">=", perfect_solution)


    for i in range(10):
        u, p, lambda0, temp_count = pi_gradient_descent(img, lambda0, perfect_solution, iter=iter)

        if temp_count != iter:
            print("Denoising time:", time() - start)
            return u

        #kein Ahnung warum das der fix ist
        #f = np.linalg.norm(u)
        #lambda0 *= perfect_solution / f

    structured = img_subtract(img, u, 1)

    textured = img_subtract(img, structured, text_factor)
    textured = convert(textured)

    print("Denoising time:", time()-start)
    return (structured, textured)

def pi_gradient_descent(img, lambda0, perfect_solution, tau = 0.25, iter = 100):
    """
    calculates pi value for a given k
    reference: "An algorithm for total variation minimization and applications" (Antonin Chambolle) Chapter 3
    :param img: gray scale CHW image
    :param lambda0: lambda factor > 0
    :param perfect_solution: sqrt(image size * image variance)
    :param tau: factor < 1/8 (ideal = 1/4)
    :param iter: number of maximum iterations
    :return: pi, last p value, lambda, number of iterations needed
    """
    start = time()

    p = np.array([ np.zeros(img.shape[1:]), np.zeros(img.shape[1:]) ])

    for i in range(iter):
        #temp values
        factored_img = img/lambda0
        diffed_p = delta_func(div_func(p) - factored_img)

        #new p value
        p_next = p + tau * diffed_p
        p_next /= 1 + tau * total_variation(diffed_p)

        #better convergence by updating lambda0 consistent
        #does not work currently
        #lambda0 = perfect_solution / np.linalg.norm(div_func(p_next))


        varition_p = np.sum(total_variation(p))
        varition_p_next = np.sum(total_variation(p_next))

        if abs(varition_p - varition_p_next) <= 0.01:
            print("calculation time for pi with ", i, "Iterations : ", time()-start)
            pi = lambda0 * div_func(p_next)
            return (pi, p_next, lambda0, i)

        #update p for next iteration
        p = p_next


    print("calculation time for pi with ", iter, "Iterations : ", time()-start)
    pi = lambda0 * div_func(p)
    return (pi, p, lambda0, iter)

def delta_func(mat):
    """
    calculates x and y gradient
    :param mat: 2 Matrices of same size
    :return: [x gradient, y gradient]
    """
    # x gradient
    kernel = np.array([[1, -1, 0], [0, 0, 0]])
    x = signal.convolve(mat[0], kernel, mode='same')
    x[:, -1] = 0

    # y gradient
    kernel = np.array([[1], [-1], [0]])
    y = signal.convolve(mat[0], kernel, mode='same')
    y[-1] = 0

    return np.array([x,y])


def div_func(p):
    """
    calculates difference between 2 matrices
    :param p: 2 Matrices of same size
    :return: difference matrix
    """
    # x gradient
    kernel = np.array([[1, -1], [0, 0]])
    x = signal.convolve(p[0], kernel, mode='same')
    x[:, -1] = (-1)*p[0][:, -2]

    # y gradient
    kernel = np.array([[1], [-1]])
    y = signal.convolve(p[1], kernel, mode='same')
    y[-1] = (-1)*p[1][-2]

    # result
    return x+y

def total_variation(mat):
    return np.sqrt(mat[0]**2 + mat[1]**2)

def img_subtract(minuend, subtrahend, factor = 0.95):
    """

    :param minuend: gray scale CHW image
    :param subtrahend: structured part of denoising
    :param factor: for impact of denoising
    :return: weighted difference
    """
    difference = minuend[0] - factor * subtrahend
    if difference.shape[0] != 1:
        difference.shape = (1, difference.shape[0], difference.shape[1])
    return difference

def convert(img):
    """
    converts image to range between 0 and 1
    :param img: gray scale CHW image
    :return: gray scale CHW image
    """
    converted = img + (-1) * img.min()
    converted /= converted.max() # oder 2
    if converted.shape[0] != 1:
        converted.shape = (1, converted.shape[0], converted.shape[1])
    return converted