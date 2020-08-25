from src.utilities.image_access import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img = open_image(r"H:\dev\opticalFlow\Prototyp\Version 2\resources\eval-twoframes\Dimetrodon\frame10.png")
    show_image(img)
    plt.show()