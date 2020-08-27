from src.horn_schunck.derivative_horn_schunck import *
from src.utilities.image_access import open_image,show_image
import numpy as np
import matplotlib.pyplot as plt

def test_get_I_x():
    img1= np.array([
        [1,2],
        [3,4]
    ])

    img2 = np.array([
        [5, 6],
        [7, 8]
    ])

    expected_result = np.array([
        [
            2-1+4-3+6-5+8-7, 0
        ],
        [
            0,0
        ]
    ],dtype=float)
    expected_result=expected_result/4
    actual_result = get_I_x(img1,img2)
    print(expected_result==actual_result)
    print("actual_result:")
    print(actual_result)

def test_derivative_test():
    img1 = open_image(r"..\..\..\resources\calibration\RradialGradient.jpg")

    img1 = img1[0]

    I_x = get_I_x(img1,img1)
    I_y = get_I_y(img1, img1)
    I_t = get_I_t(img1,img1)
    plt.figure()
    plt.imshow(I_x)
    plt.figure()
    plt.imshow(I_y)
    plt.figure()
    plt.imshow(I_t)
    plt.show()

if __name__ == '__main__':
    test_derivative_test()