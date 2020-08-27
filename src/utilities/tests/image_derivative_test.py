from src.utilities.image_derivative import *
from src.utilities.flow_field_helper import *
from src.utilities.image_access import open_image,show_image
import matplotlib.pyplot as plt

def test_differentiate_image(img):
    diff_img = differentiate_image(img)
    plt.figure()
    show_image(img)
    plt.figure()
    plt.imshow(diff_img[0,0])
    plt.figure()
    plt.imshow(diff_img[0, 1])
    plt.show()



if __name__ == '__main__':
    #img = open_image(r"..\..\..\resources\calibration\RradialGradient.jpg")
    img = open_image(r"..\..\..\resources\calibration\foatage1\frame11.jpg")
    test_differentiate_image(img)