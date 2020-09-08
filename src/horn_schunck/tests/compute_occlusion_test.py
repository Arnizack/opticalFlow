from src.horn_schunck.compute_occlusion import compute_occlusion
from src.horn_schunck.solve_layer import SolverSettings, solve_layer
from src.utilities.image_access import open_image
from src.utilities.compare_flow import compare_flow
from src.utilities.flow_field_helper import read_flow_field
from src.utilities.image_pyramid import downscale_image
import numpy as np
import matplotlib.pyplot as plt

def test_occlusion(img1, img2):
    settings = SolverSettings()
    img1 = downscale_image(img1,0.2)
    img2 = downscale_image(img2, 0.2)
    init_flow = np.zeros(shape=(2, img1.shape[1], img1.shape[2]))

    flow = solve_layer(img1, img2,init_flow,settings)

    occlusion = compute_occlusion(img1,img2,flow)

    return occlusion

def dimetrodon():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11-gray.png")
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")
    return img1,img2,ref_flow

if __name__ == '__main__':
    img1, img2, ref_flow = dimetrodon()
    occlusion = test_occlusion(img1,img2)
    plt.imshow(occlusion)
    plt.show()

