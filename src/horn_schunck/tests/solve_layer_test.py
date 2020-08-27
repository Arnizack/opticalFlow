from src.horn_schunck.solve_layer import setup_linear_system,SolverSettings,solve_layer
import numpy as np
import matplotlib.pyplot as plt
from time import time

from src.utilities.image_access import open_image, show_image
from src.utilities.image_derivative import differentiate_image
from  src.utilities.flow_field_helper import show_flow_field,read_flow_field
from src.utilities.image_pyramid import downscale_image



def test_layer1(img1,img2):


    img1 = downscale_image(img1, 0.1)
    img2 = downscale_image(img2, 0.1)

    der_img2 = differentiate_image(img2)
    plt.imshow(der_img2[0,0])
    plt.figure()
    plt.imshow(der_img2[0, 1])
    plt.show()
    settings = SolverSettings()
    settings.alpha=15
    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]))

    width = img2.shape[2]
    height = img2.shape[1]
    start = time()
    for iter in range(1):
        flow = solve_layer(img1,img2,der_img2,init_flow,settings)
        init_flow+=flow
    print("Solve Layer time: ", time() - start)
    plt.show()
    show_flow_field(np.array([init_flow[1],init_flow[0]]),height,width)
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")
    show_flow_field(ref_flow,ref_flow.shape[1],ref_flow.shape[2])
    plt.show()

def test_layer2():
    img1 = open_image(r"..\..\..\resources\calibration\foatage1\frame10.jpg")
    img2 = open_image(r"..\..\..\resources\calibration\foatage1\frame11.jpg")

    show_image(img1)


    #img1 = downscale_image(img1, 0.05)
    #img2 = downscale_image(img2, 0.05)

    der_img2 = differentiate_image(img2)
    settings = SolverSettings()
    settings.alpha=15
    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]))

    width = img2.shape[2]
    height = img2.shape[1]
    start = time()
    flow = solve_layer(img1,img2,der_img2,init_flow,settings)
    print("Solve Layer time: ",time()-start)
    show_flow_field(np.array([flow[1],flow[0]]),height,width)
    plt.show()

if __name__ == '__main__':
    #test_setup_linear_system()
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11-gray.png")
    img1 = img1[[0]]
    img2 = img2[[0]]
    #img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    #img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")
    test_layer1(img1,img2)
