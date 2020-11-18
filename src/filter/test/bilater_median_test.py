#from src.filter.bilater_median import bilateral_median_filter
import src.filter.cython.bilateral_median as c_filter
from src.filter.bilater_median import bilateral_median_filter
from src.utilities.compute_occlusion import compute_occlusion, compute_occlusion_log
from src.horn_schunck.solve_layer import SolverSettings, solve_layer
from src.utilities.image_access import open_image
from src.utilities.image_pyramid import downscale_image
from src.utilities.flow_field_helper import show_flow_field

import matplotlib.pyplot as plt
import numpy as np
from time import time

def dimetrodon():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11-gray.png")

    return img1,img2

def grove3():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Grove3\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Grove3\frame11.png")
    return img1,img2

def test_bilateral_median_filter_bug():
    img1 = [
           [[0.29565689, 0.29208999, 0.29268468, 0.11084937, 0.11727952],
            [0.32344712, 0.2866968 , 0.23681973, 0.32194382, 0.27420041],
            [0.37133725, 0.54234784, 0.34309348, 0.27406237, 0.30526862],
            [0.49302601, 0.69430977, 0.42510721, 0.29044928, 0.26380874],
            [0.33955504, 0.56177763, 0.55550258, 0.16565074, 0.20440195]],
           [[0.28586201, 0.27651482, 0.26964234, 0.11071693, 0.11417289],
            [0.30987525, 0.26955097, 0.2361963 , 0.41282661, 0.32410419],
            [0.45136255, 0.68128379, 0.39008932, 0.25955672, 0.29906917],
            [0.51197335, 0.83576093, 0.47369368, 0.28479119, 0.26585297],
            [0.31848839, 0.69716167, 0.64441143, 0.1653485 , 0.2003209 ]],
           [[0.24026179, 0.21938395, 0.19907541, 0.10780376, 0.11252355],
            [0.25992555, 0.21771755, 0.16778322, 0.21649854, 0.19063535],
            [0.25470176, 0.38945105, 0.28555226, 0.25009905, 0.28901243],
            [0.37070551, 0.54087005, 0.36004111, 0.28544221, 0.26303182],
            [0.25687337, 0.41850402, 0.45295518, 0.15633663, 0.18173423]]
    ]

    img2 = [[[0.30258176,  0.29778858,  0.23746149, 0.17956648, 0.10750849],
             [0.17136594,  0.26427312,  0.23463514, 0.24429038, 0.26129225],
             [0.26609797,  0.33726028,  0.40078054, 0.20971017, 0.35413609],
             [0.35563237,  0.6617327 ,  0.72374927, 0.27423921, 0.28556916],
             [0.29274756,  0.31896984,  0.64299549, 0.2096851 , 0.20299824]],
            [[0.29963331,  0.28106231,  0.2307286 , 0.16876867, 0.10505913],
             [0.1691965 ,  0.25435223,  0.21587643, 0.29787555, 0.30102485],
             [0.24380409,  0.37804251,  0.4735718 , 0.20541528, 0.34394449],
             [0.37228684,  0.82047863,  0.85912182, 0.26857558, 0.28616473],
             [0.28293057,  0.36570935,  0.77339228, 0.21757725, 0.19674616]],
           [[0.26668561 , 0.21882525 , 0.19371361 ,0.1369206  ,0.10507734],
            [0.1574743  , 0.21393293 , 0.16718274 ,0.17693323 ,0.18536691],
            [0.1899262  , 0.22700855 , 0.27915755 ,0.18413194 ,0.31119419],
            [0.31387883 , 0.49329326 , 0.57737184 ,0.26782394 ,0.28158657],
            [0.25680199 , 0.2351433  , 0.49752528 ,0.20350419 ,0.19167993]]]

    occlusion = [
        [ -0.21855347 , -0.33513958 , -4.06874679 , -1.276615  , -2.40064262],
        [ -0.51975004 , -0.05620835 , -2.55543799 , -0.27774014,-16.77294452],
        [-11.90939081 , -4.18100178 , -1.96762826 , -1.54728571,-21.45607843],
        [-21.91492715 , -2.63107706 , -5.78915986 , -0.08737177,-21.96253488],
        [ -5.94522974 , -0.18044585 ,-24.16026907 , -0.32021952,-12.94242146]]



    flow = [
               [
                    [0.1987233 , 0.20293018, 0.15007553, 0.1568506 , 0.16885778],
                    [0.11904475, 0.11904475, 0.10293369, 0.1568506 , 0.21059042],
                    [0.10001883, 0.23119104, 0.11057286, 0.20705959, 0.23940435],
                    [0.15555215, 0.18459567, 0.18459567, 0.20705959, 0.25586906],
                    [0.18459567, 0.1249193 , 0.08823799, 0.1880285 , 0.23940435]
                ],
                [
                    [0.82124794, 0.67275935, 0.7600328 , 0.7170666 , 0.61784625],
                    [0.5524652 , 0.39323354, 0.4877802 , 0.60096055, 0.6399983 ],
                    [0.37989205, 0.65654135, 0.4364441 , 0.6120315 , 0.6120315 ],
                    [0.5367303 , 0.45425504, 0.6120315 , 0.65129936, 0.6820165 ],
                    [0.62003505, 0.38508236, 0.44015813, 0.6262438 , 0.64525753]
                ]
    ]

    flow=np.array(flow)
    img1 = np.array(img1)
    img2 = np.array(img2)
    occlusion = np.array(occlusion)
    """
    start_y = 2
    end_y = 4
    start_x = 0
    end_x = 2

    flow = flow[:,start_y:end_y,start_x:end_x]
    img1 = img1[:,start_y:end_y,start_x:end_x]
    img2 = img2[:, start_y:end_y, start_x:end_x]
    occlusion = occlusion[start_y:end_y, start_x:end_x]
    """



    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]), dtype=np.double)
    flow = bilateral_median_filter(flow.astype(np.double), occlusion.astype(np.double), init_flow.astype(np.double),
                                   img1.astype(np.double),
                                   weigth_auxiliary=1, weigth_filter=1, sigma_distance=1, sigma_color=1, filter_size=3)


    return flow

def test_bilateral_median_filter_bug2():
    img1 = [
           [
            [0.49302601, 0.69430977],
            [0.33955504, 0.56177763]],
           [
            [0.51197335, 0.83576093],
            [0.31848839, 0.69716167]],
           [
            [0.37070551, 0.54087005],
            [0.25687337, 0.41850402]]
    ]

    img2 = [[
             [0.35563237,  0.6617327 ],
             [0.29274756,  0.31896984]],
            [
             [0.37228684,  0.82047863],
             [0.28293057,  0.36570935]],
           [
            [0.31387883 , 0.49329326 ],
            [0.25680199 , 0.2351433  ]]]

    occlusion = [

        [-21.91492715 , -2.63107706 ],
        [ -5.94522974 , -0.18044585 ]]



    flow = [
               [

                    [0.15555215, 0.18459567],
                    [0.18459567, 0.1249193 ]
                ],
                [

                    [0.5367303 , 0.45425504],
                    [0.62003505, 0.38508236]
                ]
    ]

    flow=np.array(flow)
    img1 = np.array(img1)
    img2 = np.array(img2)
    occlusion = np.array(occlusion)
    """
    start_y = 2
    end_y = 4
    start_x = 0
    end_x = 2

    flow = flow[:,start_y:end_y,start_x:end_x]
    img1 = img1[:,start_y:end_y,start_x:end_x]
    img2 = img2[:, start_y:end_y, start_x:end_x]
    occlusion = occlusion[start_y:end_y, start_x:end_x]"""

    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]), dtype=np.double)
    flow = bilateral_median_filter(flow.astype(np.double), occlusion.astype(np.double), init_flow.astype(np.double),
                                   img1.astype(np.double),
                                   weigth_auxiliary=2, weigth_filter=6, sigma_distance=5, sigma_color=4,
                                   filter_size=3)

    return flow


def test_bilateral_median_filter_color():
    img1 = [
           [
                [1, 0],
                [1, 0]],
           [
               [1, 0],
               [1, 0]],
           [
               [1, 0],
               [1, 0]]
    ]



    occlusion = [

        [0, 0 ],
        [ 0 , 0 ]]



    flow = [
               [

                    [1, 0],
                    [1, 0]
                ],
                [

                    [1 , 0],
                    [1, 0]
                ]
    ]

    flow=np.array(flow)
    img1 = np.array(img1)
    occlusion = np.array(occlusion)
    """
    start_y = 2
    end_y = 4
    start_x = 0
    end_x = 2

    flow = flow[:,start_y:end_y,start_x:end_x]
    img1 = img1[:,start_y:end_y,start_x:end_x]
    img2 = img2[:, start_y:end_y, start_x:end_x]
    occlusion = occlusion[start_y:end_y, start_x:end_x]"""

    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]), dtype=np.double)
    flow = bilateral_median_filter(flow.astype(np.double), occlusion.astype(np.double), init_flow.astype(np.double),
                                   img1.astype(np.double),
                                   weigth_auxiliary=1, weigth_filter=1, sigma_distance=1, sigma_color=1, filter_size=3)

    return flow

def test_bilateral_median_filter(img1,img2):
    settings = SolverSettings()
    settings.median_filter_size = -1
    img1 = downscale_image(img1, 0.1)
    img2 = downscale_image(img2, 0.1)
    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]),dtype = np.double)

    flow = solve_layer(img1, img2, init_flow, settings)

    #flow = flow[:,30:35,50:55]
    #img1 = img1[:,30:35,50:55]
    #img2 = img2[:, 30: 35, 50:55]

    plt.title("Flow")
    show_flow_field(flow, flow.shape[2], flow.shape[1])
    plt.show()


    occlusion = compute_occlusion_log(img1, img2, flow)

    plt.title("Occlusion")
    plt.imshow(np.exp(occlusion))
    plt.show()

    start = time()

    flow = bilateral_median_filter(flow.astype(np.double),occlusion.astype(np.double),init_flow.astype(np.double),img1.astype(np.double),
                                   weigth_auxiliary=2, weigth_filter=6, sigma_distance=5, sigma_color=4,
                                   filter_size=3)
    c_flow = c_filter.bilateral_median_filter(flow.astype(np.double), occlusion.astype(np.double),
                                              init_flow.astype(np.double),
                                              img1.astype(np.double),
                                              weigth_auxiliary=2, weigth_filter=6, sigma_distance=5, sigma_color=4,
                                              filter_size=3)

    print("Bilateral Median Filter Time: ",time()-start)

    return flow,c_flow

if __name__ == '__main__':
    img1,img2 = grove3()
    flow = test_bilateral_median_filter_bug2()

    #flow[0][flow[0]>7]=0
    #flow[1][flow[1] > 7] = 0
    plt.title("Final")
    print(flow.flatten())
    show_flow_field(flow,flow.shape[2],flow.shape[1],mode="RGB")
    show_flow_field(flow, flow.shape[2], flow.shape[1], mode="Split")


    plt.show()
