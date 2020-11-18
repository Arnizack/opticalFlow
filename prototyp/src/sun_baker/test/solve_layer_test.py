from src.utilities.image_pyramid import downscale_image
from src.sun_baker.solver_settings import SolverSettings
import numpy as np
from time import time
import matplotlib.pyplot as plt
from src.sun_baker.solve_layer import solve_layer
from src.utilities.penalty_functions.SquaredPenalty import SquaredPenalty
from src.utilities.penalty_functions.MixPenalty import MixPenalty
from src.utilities.penalty_functions.GeneralizedCharbonnierPenalty import GeneralizedCharbonnierPenalty
from src.utilities.image_access import open_image
from src.utilities.flow_field_helper import read_flow_field
from src.utilities.compare_flow import compare_flow
from src.utilities.flow_field_helper import show_flow_field
from src.sun_baker.test.test_condition import test_condition

def test_layer1(img1,img2):



    settings = SolverSettings()
    settings.alpha=10/200
    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]))

    width = img2.shape[2]
    height = img2.shape[1]
    start = time()
    plt.figure()
    penalty_func_1 = SquaredPenalty()
    penalty_func_2 = GeneralizedCharbonnierPenalty()


    penalty_func = MixPenalty(penalty_func_1,penalty_func_2,0)

    gnc_steps = 3

    for iter,lambda_k in zip(range(gnc_steps),[200/255,200/255,200/255]):
        print("GNC: ",iter)
        if(gnc_steps>1):
            mix_factor = iter/(gnc_steps-1)
        else:
            mix_factor = 0
        settings.weight_kernel=lambda_k
        penalty_func.mix_factor = mix_factor
        for iter2 in range(3):
            init_flow = solve_layer(img1,img2,init_flow, penalty_func,settings)
        show_flow_field(init_flow,width,height)



    return init_flow

if __name__ == '__main__':
    #test_setup_linear_system()
    img1 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    img2 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")
    #img1 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")
    #img2 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame11-gray.png")
    #img1 = open_image(r"..\..\..\..\resources\eval-twoframes\syntetisch\frame10.jpg")
    #img2 = open_image(r"..\..\..\..\resources\eval-twoframes\syntetisch\frame11.jpg")

    #img1 = img1[[0]]
    #img2 = img2[[0]]
    #img1 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    #img2 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")

    img1 = downscale_image(img1, 0.2)
    img2 = downscale_image(img2, 0.2)

    computed_flow = test_layer1(img1,img2)

    ref_flow = read_flow_field(r"..\..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")

    settings = SolverSettings()

    penalty_func = GeneralizedCharbonnierPenalty()
    error = test_condition(img1, img2, computed_flow, penalty_func, settings)
    plt.imshow(error)

    compare_flow(computed_flow,ref_flow,img1,img2,plot=True)
