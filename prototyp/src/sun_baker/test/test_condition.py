from src.utilities.image_derivative import differentiate_image
from src.sun_baker.solver_settings import SolverSettings
from src.utilities.penalty_functions.IPenalty import IPenalty
from src.utilities.warp_grid import warp_image
from src.utilities.color2grayscale import color2grayscale
import numpy as np
from src.utilities.flow_field_helper import read_flow_field
from src.utilities.image_access import open_image
from src.utilities.penalty_functions.GeneralizedCharbonnierPenalty import GeneralizedCharbonnierPenalty
import matplotlib.pyplot as plt

def test_condition(first_image : np.ndarray, second_image : np.ndarray, flow : np.ndarray,penalty_func : IPenalty, settings : SolverSettings):
    """

    :param first_image: (Colorchannel, Height, Width)
    :param second_image: (Colorchannel, Height, Width)
    :param flow: (2,Height, Width)
    :return: np.ndarray (Height, Width)
    """

    second_image_warped = warp_image(second_image,flow)

    gray_second_img_warped = color2grayscale(second_image_warped)
    gray_first_img = color2grayscale(first_image)


    difference = penalty_func.get_value_at(gray_first_img-gray_second_img_warped)

    derivative = differentiate_image(flow)

    error = difference + settings.weight_kernel*(derivative[0][0]**2+derivative[1][1]**2)

    return error

if __name__ == '__main__':
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")

    settings = SolverSettings()
    settings.weight_kernel=180/255
    penalty_func = GeneralizedCharbonnierPenalty()

    error = test_condition(img1,img2,ref_flow,penalty_func,settings)
    print("Mean error: ",np.median(error))
    plt.imshow(error)
    plt.show()