from src.horn_schunck.solve_layer import setup_linear_system,SolverSettings,solve_layer
import numpy as np
import matplotlib.pyplot as plt
from time import time

from src.utilities.image_access import open_image, show_image
from src.utilities.image_derivative import differentiate_image
from  src.utilities.flow_field_helper import show_flow_field,read_flow_field,show_flow_field_arrow
from src.utilities.image_pyramid import downscale_image
from src.utilities.warp_grid import warp_image
from scipy.signal import medfilt2d

def test_layer1(img1,img2):


    img1 = downscale_image(img1, 0.2)
    img2 = downscale_image(img2, 0.2)

    der_img2 = differentiate_image(img2)
    der_img1 = differentiate_image(img1)

    settings = SolverSettings()
    settings.alpha=15
    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]))

    width = img2.shape[2]
    height = img2.shape[1]
    start = time()
    plt.figure()
    for iter in range(10):
        flow = solve_layer(img1,img2,der_img1,der_img2,init_flow,settings)
        flow[0] = medfilt2d(flow[0],5)
        flow[1] = medfilt2d(flow[1], 5)
        init_flow+=flow
    print("Solve Layer time: ", time() - start)
    plt.show()
    plt.title("Y Flow")
    plt.imshow(init_flow[0])
    plt.figure()
    plt.title("X Flow")
    plt.imshow(init_flow[1])
    plt.show()

    print("Warped result")

    img2_warped = warp_image(img2, np.array([init_flow[0],init_flow[1]]))
    img1_warped = warp_image(img1, np.array([init_flow[0], init_flow[1]]))
    show_image(img1)
    plt.title("img1")
    plt.figure()
    show_image(img2_warped)
    plt.title("Warped Img2")
    plt.figure()
    diff = img1-img2_warped
    show_image(diff)
    plt.title("img1-img2_warped")
    plt.figure()
    show_image(img1-img2)
    plt.title("img1-img2")

    print("img1-img2_warped mean error: ",np.mean(diff**2))
    plt.show()
    plt.imshow(img1[0])
    show_flow_field_arrow(np.array([init_flow[1],init_flow[0]]),width,height)
    #plt.show()
    show_flow_field(np.array([init_flow[1],init_flow[0]]),height,width)


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
    #img1 = open_image(r"..\..\..\resources\eval-twoframes\syntetisch\frame10.jpg")
    #img2 = open_image(r"..\..\..\resources\eval-twoframes\syntetisch\frame11.jpg")

    img1 = img1[[0]]
    img2 = img2[[0]]
    #img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    #img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")
    show_image(img1)
    plt.show()
    test_layer1(img1,img2)

    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")
    show_flow_field(ref_flow, ref_flow.shape[1], ref_flow.shape[2])
    plt.show()