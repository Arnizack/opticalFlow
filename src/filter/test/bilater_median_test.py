#from src.filter.bilater_median import bilateral_median_filter
from src.filter.cython.bilateral_median import bilateral_median_filter
from src.horn_schunck.compute_occlusion import compute_occlusion
from src.horn_schunck.solve_layer import SolverSettings, solve_layer
from src.utilities.image_access import open_image
from src.utilities.compare_flow import compare_flow
from src.utilities.flow_field_helper import read_flow_field
from src.utilities.image_pyramid import downscale_image
from src.utilities.flow_field_helper import show_flow_field

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from time import time

def dimetrodon():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11-gray.png")

    return img1,img2

def test_bilateral_median_filter(img1,img2):
    settings = SolverSettings()
    img1 = downscale_image(img1, 0.2)
    img2 = downscale_image(img2, 0.2)
    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]))

    flow = solve_layer(img1, img2, init_flow, settings)

    show_flow_field(flow, flow.shape[2], flow.shape[1])
    plt.show()

    occlusion = compute_occlusion(img1, img2, flow)

    start = time()

    flow = bilateral_median_filter(flow,occlusion,init_flow,img1,weigth_auxiliary=5,weigth_filter=2)

    print("Bilateral Median Filter Time: ",time()-start)

    return flow

if __name__ == '__main__':
    img1,img2 = dimetrodon()
    flow = test_bilateral_median_filter(img1,img2)

    show_flow_field(flow,flow.shape[2],flow.shape[1])
    plt.show()
