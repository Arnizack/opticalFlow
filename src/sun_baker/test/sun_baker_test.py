from src.sun_baker.sun_baker import sun_baker_optical_flow
from src.sun_baker.solver_settings import SolverSettings
from src.utilities.image_access import open_image
from src.utilities.compare_flow import compare_flow
from src.utilities.flow_field_helper import read_flow_field
from src.sun_baker.test.test_condition import test_condition
import numpy as np
from src.sun_baker.solver_settings import SolverSettings
from src.utilities.penalty_functions.GeneralizedCharbonnierPenalty import GeneralizedCharbonnierPenalty
import matplotlib.pyplot as plt

def test_sun_baker(img1,img2):
    settings = SolverSettings()
    flow = sun_baker_optical_flow(img1,img2)

    return flow

def dimetrodon():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")
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

def Grove2():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Grove2\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Grove2\frame11.png")
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Grove2\flow10.flo")
    return img1, img2, ref_flow

def Urban2():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Urban2\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Urban2\frame11.png")
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Urban2\flow10.flo")
    return img1, img2, ref_flow

def Urban3():
    img1 = open_image(r"..\..\..\resources\eval-twoframes\Urban3\frame10.png")
    img2 = open_image(r"..\..\..\resources\eval-twoframes\Urban3\frame11.png")
    ref_flow = read_flow_field(r"..\..\..\resources\eval-twoframes-groundtruth\Urban3\flow10.flo")
    return img1, img2, ref_flow

if __name__ == '__main__':
    img1, img2, ref_flow = Grove2()
    computed_flow = test_sun_baker(img1,img2)
    error = test_condition(img1,img2,computed_flow,GeneralizedCharbonnierPenalty(),SolverSettings())

    plt.title("Hypothese")
    plt.imshow(error)
    plt.figure()


    compare_flow(computed_flow, ref_flow, img1, img2, plot=True,arrows=True)