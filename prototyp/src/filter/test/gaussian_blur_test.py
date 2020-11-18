from src.utilities.image_access import open_image,show_image
from src.filter.gaussian_blur import gaussian_blur_matrix, gaussian_blur_flow
from src.utilities.flow_field_helper import show_flow_field,read_flow_field
from src.utilities.image_pyramid import downscale_image
from src.utilities.compare_flow import scale_flow
import matplotlib.pyplot as plt
from scipy import signal

import numpy as np
import math

def dimetrodon():
    img = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")

    return img

def test_mat():
    img = dimetrodon()

    img = gaussian_blur_matrix(-img[0], 1)
    plt.imshow(img)
    plt.show()

def test_flow():
    flow = read_flow_field(r"..\..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")



    #flow = scale_flow(flow,int(flow.shape[2]/80),int(flow.shape[1]/80))

    flow_ = np.array(
        [
            [
                [1, 2],
                [5, 6]
            ],
            [
                [8, 7],
                [4, 3]
            ]
        ]
    )

    flow_ = np.full(shape=(2,4,4),fill_value=1)

    print(flow)
    show_flow_field(flow, flow.shape[2], flow.shape[1])
    plt.show()

    flow_blur = gaussian_blur_flow(flow,1)
    show_flow_field(flow_blur, flow.shape[2], flow.shape[1])
    plt.show()

if __name__ == '__main__':
    test_flow()
