import numpy as np
import matplotlib.pyplot as plt
from src.sun_baker.setup_linear_system.setup_linear_system import setup_linear_system
from src.utilities.penalty_functions.SquaredPenalty import SquaredPenalty
from src.utilities.color2grayscale import color2grayscale
from src.utilities.image_access import open_image
from src.sun_baker.derivative_sun import derivative_sun
import src.horn_schunck.setup_linear_system as HS
from src.horn_schunck.solver_settings import SolverSettings as HS_Settings
from src.utilities.image_pyramid import downscale_image


def setup_linear_system_test1():
    width = 4
    height = 5

    I_x = np.full(shape=(height * width),fill_value=1)
    I_y = np.full(shape=(height * width), fill_value=2/10)
    I_t = np.full(shape=(height * width), fill_value=3/10)

    kernel = np.array(
        [[1, 2, 1],
         [2, -12, 2],
         [1, 2, 1]] ) / 12

    guess_vu = np.full(shape=(2 * height * width), fill_value=4/10)
    relax_vu = np.full(shape=(2 * height * width), fill_value=5/10)
    lambda_r = 1
    lambda_k = 1

    A,b = setup_linear_system(I_x, I_y, I_t, guess_vu, relax_vu,
                            kernel, width, height, lambda_r , lambda_k ,
                            SquaredPenalty())
    dense_A = A.todense()

    plt.matshow(dense_A)
    plt.show()

def setup_linear_system_test2(img1,img2):
    img1 = downscale_image(img1, 0.02)
    img2 = downscale_image(img2, 0.02)
    width = img1.shape[2]
    height = img1.shape[1]

    img1_gray =color2grayscale(img1)
    img2_gray = color2grayscale(img2)

    I_x, I_y, I_t = derivative_sun(img1_gray, img2_gray)
    I_x.shape = (height * width)
    I_y.shape = (height * width)
    I_t.shape = (height * width)

    kernel = np.array(
        [[1, 2, 1],
         [2, -12, 2],
         [1, 2, 1]]) / 12

    guess_vu = np.full(shape=(2 * height * width), fill_value=0)
    relax_vu = np.full(shape=(2 * height * width), fill_value=0)
    lambda_r = 0.01
    lambda_k = 20/100

    A_sun,b_sun = setup_linear_system(I_x, I_y, I_t, guess_vu, relax_vu,
                            kernel, width, height, lambda_r , lambda_k**2 ,
                            SquaredPenalty())
    dense_A_sun = A_sun.todense()

    img1_HS = np.array([img1_gray])
    img2_HS = np.array([img2_gray])

    settings_hs = HS_Settings()

    settings_hs.alpha = lambda_k

    A_hs, b_hs = HS.setup_linear_system(img1_HS,img2_HS,settings_hs)

    dense_A_hs = A_hs.todense()

    plt.matshow(dense_A_sun)
    plt.matshow(dense_A_hs)

    plt.matshow(b_sun.reshape(height*2,width))
    plt.matshow(b_hs.reshape(height*2,width))

    plt.show()


if __name__ == '__main__':
    img1 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    img2 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")
    setup_linear_system_test2(img1,img2)