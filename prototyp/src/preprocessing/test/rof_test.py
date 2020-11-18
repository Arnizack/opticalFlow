from src.utilities.image_access import open_image
from src.horn_schunck.tests.horn_schunck_test import dimetrodon,grove3, RubberWhale,grove2
from src.preprocessing.rof import denoising_chambolle, div_func, delta_func, pi_gradient_descent, img_subtract
from src.utilities.image_access import show_image
from src.utilities.color2grayscale import color2grayscale
import matplotlib.pyplot as plt

import numpy as np

def denoising_chambolle_test():

    img1, img2, ref_flow = grove2()

    if img1.shape[0] != 1:
        img1 = color2grayscale(img1)

    print(img1.shape)

    std = img1.std()

    structured, textured = denoising_chambolle(img1, lambda0=0.125, std_dev=0.5,iter=20)

    diff = img1 - structured

    statistics(img1, "Image")
    statistics(structured, "Denoised")
    statistics(diff, "Difference")
    statistics(textured, "Textured")

    plt.figure()
    #img1.shape = (1, img1.shape[0], img1.shape[1])
    plt.imshow(img1)
    plt.title("Standard")

    plt.figure()
    #structured.shape = (1, structured.shape[0], structured.shape[1])
    plt.imshow(structured)
    plt.title("Denoised")

    plt.figure()
    #textured.shape = (1, textured.shape[0], textured.shape[1])
    plt.imshow(textured/textured.std())
    plt.title("Textured")

    plt.show()


def delta_func_test():
    mat = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    control_x = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    control_y = np.array([[3, 3, 3], [3, 3, 3], [0, 0, 0]])

    x, y = delta_func(mat)

    if (x == control_x).all() != True:
        print("div_func: X gradient different")
    elif (y == control_y).all() != True:
        print("div_func: Y gradient different")
    else:
        print("div_func: X and Y correct")

def div_func_test():
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    control_x = np.array([[1, 1, -2], [4, 1, -5], [7, 1, -8]])
    control_y = np.array([[1, 2, 3], [3, 3, 3], [-4, -5, -6]])

    control = control_x + control_y

    x = div_func((mat, mat))

    if (x == control).all() != True:
        print("delta_func: different")
    else:
        print("delta_func: correct")

def pi_gradient_descent_test():
    img1, img2, ref_flow = dimetrodon()
    lambda0 = 1

    x, i = pi_gradient_descent(img1, lambda0)
    print(x)
    print(i)

def statistics(mat, title = ""):
    print(title,":\t Mean", mat.mean(), "Min", mat.min(), "Max", mat.max(), "std", mat.std(), "var", mat.var())

if __name__ == '__main__':
    #div_func_test()
    #delta_func_test()

    #pi_gradient_descent_test()

    denoising_chambolle_test()