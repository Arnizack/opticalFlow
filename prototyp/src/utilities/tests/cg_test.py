from src.horn_schunck.solve_layer import setup_linear_system,SolverSettings,solve_layer
import numpy as np
import matplotlib.pyplot as plt
from time import time

from src.utilities.image_access import open_image, show_image
from src.utilities.image_derivative import differentiate_image
from  src.utilities.flow_field_helper import show_flow_field,read_flow_field,show_flow_field_arrow
from src.utilities.image_pyramid import downscale_image
from src.utilities.warp_grid import warp_image

from src.utilities.compare_flow import compare_flow

def test_layer1(img1,img2, solver_settter = "cg"):


    img1 = downscale_image(img1, 1)
    img2 = downscale_image(img2, 1)

    der_img2 = differentiate_image(img2)
    der_img1 = differentiate_image(img1)

    settings = SolverSettings()
    settings.alpha=15/200
    settings.solver = solver_settter
    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]))

    width = img2.shape[2]
    height = img2.shape[1]
    start = time()
    plt.figure()
    for iter in range(3):
        flow = solve_layer(img1,img2,init_flow,settings)

        init_flow=flow
        print("")
    return init_flow

def compare(flow_scipy, flow_own):
    if flow_scipy.shape != flow_own.shape:
        print("Different Shapes")
    print("Scipy:")
    stats(flow_scipy)
    print("Own:")
    stats(flow_own)
    print("Difference:")
    err = flow_own - flow_scipy
    stats(err)

def stats(mat):
    print("Min: ", mat.min(), ", Max: ", mat.max(), ", Mean: ", mat.mean())

if __name__ == '__main__':
    #test_setup_linear_system()
    img1 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    img2 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")
    img1 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")
    img2 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame11-gray.png")
    #img1 = open_image(r"..\..\..\..\resources\eval-twoframes\syntetisch\frame10.jpg")
    #img2 = open_image(r"..\..\..\..\resources\eval-twoframes\syntetisch\frame11.jpg")

    #img1 = img1[[0]]
    #img2 = img2[[0]]
    #img1 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    #img2 = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")

    computed_flow = test_layer1(img1,img2, "cg")

    computed_flow_own = test_layer1(img1,img2, "cg_own")

    ref_flow = read_flow_field(r"..\..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")

    compare(computed_flow, computed_flow_own)

    #compare_flow(computed_flow, computed_flow_peer, img1, img2, plot=True)
    #compare_flow(computed_flow,ref_flow,img1,img2,plot=True)