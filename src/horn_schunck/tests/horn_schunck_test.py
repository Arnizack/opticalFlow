from src.horn_schunck.horn_schunck import compute_optical_flow_HS, SolverSettings
from src.utilities.image_access import open_image
from src.utilities.compare_flow import compare_flow
from src.utilities.flow_field_helper import read_flow_field
from src.utilities.scale_np_array_in_range import scale_np_array_in_range
import numpy as np
from src.utilities.image_access import show_image
import matplotlib.pyplot as plt

def test_horn_schunck(img1,img2):
    settings = SolverSettings()
    flow = compute_optical_flow_HS(img1,img2)

    return flow

def dimetrodon():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11-gray.png")
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Dimetrodon\flow10.flo")
    return img1,img2,ref_flow

def grove3():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Grove3\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Grove3\frame11.png")
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Grove3\flow10.flo")
    return img1,img2,ref_flow

def dogdance():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\DogDance\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\DogDance\frame11.png")

    ref_flow = np.zeros(shape=(2,img1.shape[1],img1.shape[2]))
    return img1,img2,ref_flow

def MiniCooper():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\MiniCooper\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\MiniCooper\frame11.png")

    ref_flow = np.zeros(shape=(2,img1.shape[1],img1.shape[2]))
    return img1,img2,ref_flow

def RubberWhale():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\RubberWhale\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\RubberWhale\frame11.png")

    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\RubberWhale\flow10.flo")
    return img1, img2, ref_flow

def grove2():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Grove2\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Grove2\frame11.png")
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Grove2\flow10.flo")
    return img1, img2, ref_flow

if __name__ == '__main__':
    img1, img2, ref_flow = RubberWhale()
    #img1 = denoising_chambolle_color_img(img1)
    #img2 = denoising_chambolle_color_img(img2)
    img1 = scale_np_array_in_range(img1,0,1)
    img2 = scale_np_array_in_range(img2,0,1)

    show_image(img1)
    plt.figure()
    show_image(img2)
    plt.show()
    computed_flow = test_horn_schunck(img1,img2)



    compare_flow(computed_flow, ref_flow, img1, img2, plot=True,arrows=True)