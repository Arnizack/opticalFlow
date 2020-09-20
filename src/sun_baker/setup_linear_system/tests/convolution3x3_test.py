from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from src.sun_baker.setup_linear_system.convolution3x3 import convolution3x3

def test_conv():
    expected_A = [
        [5, 6, 0, 8, 9, 0, 0, 0, 0, 0, 0, 0],
        [4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0],
        [0, 4, 5, 0, 7, 8, 0, 0, 0, 0, 0, 0],

        [2, 3, 0, 5, 6, 0, 8, 9, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0],
        [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 0, 0],

        [0, 0, 0, 2, 3, 0, 5, 6, 0, 8, 9, 0],
        [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [0, 0, 0, 0, 1, 2, 0, 4, 5, 0, 7, 8],

        [0, 0, 0, 0, 0, 0, 2, 3, 0, 5, 6, 0],
        [0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
        [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 4, 5]
    ]
    kernel = [[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]

    width = 3
    height = 4

    A = convolution3x3(kernel,width,height)

    actual_A = A.todense()

    plt.title("Actual A")
    plt.matshow(actual_A)
    plt.figure()
    plt.title("Expected A")
    plt.matshow(expected_A)
    plt.show()

if __name__ == '__main__':
    test_conv()