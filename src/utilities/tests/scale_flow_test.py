from src.utilities.scale_flow_field import *
from src.utilities.flow_field_helper import show_flow_field,read_flow_field

import numpy as np

import matplotlib.pyplot as plt

def test_upscale_visual():
    flow = np.array(
        [
            [
                [1,2,3,4],
                [5,6,7,8],
                [9,10,11,12],
                [13,14,15,16]
            ],
            [
                [8,7,6,5],
                [4,3,2,1],
                [12,11,10,9],
                [13, 14, 15, 16]
            ]
        ]
    )
    factor = 1.5
    width = flow.shape[2]
    height = flow.shape[1]


    plt.title("Original Flow")
    show_flow_field(flow,width,height)



    scaled_flow = upscale_flow(flow,int(width*factor),int(height*factor))

    plt.figure()
    plt.title("Original Flow")
    show_flow_field(scaled_flow,int(width*factor),int(height*factor))
    plt.show()

def test_downscale_visual(flow):
    flow = down_scale_flow(flow,flow.shape[2]/6,flow.shape[1]/6)
    width = flow.shape[2]
    height = flow.shape[1]
    show_flow_field(flow,width,height)
    plt.show()

if __name__ == '__main__':
    flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")
    test_downscale_visual(flow)
