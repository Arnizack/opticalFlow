from src.utilities.image_pyramid import *
from src.utilities.image_access import open_image,show_image
import numpy as np
import matplotlib.pyplot as plt

def test_downsacle(img):

    plt.figure()
    show_image(img)

    img2R = downscale(img[0], 0.5)
    img2G = downscale(img[1], 0.5)
    img2B = downscale(img[1], 0.5)
    img2 = np.array([img2R, img2G, img2B])
    plt.figure()
    show_image(img2)
    plt.show()

def test_image_pyramid(img,factors):
    pyramid = create_image_pyramid(img,factors)
    for level in pyramid:
        plt.figure()
        show_image(level)
    plt.show()

if __name__ == '__main__':
    img = open_image(r"..\..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    test_image_pyramid(img,[0.8,0.5,0.5])
