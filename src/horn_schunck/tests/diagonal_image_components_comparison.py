import matplotlib.pyplot as plt
import numpy as np
from src.utilities.image_access import open_image,show_image
from src.horn_schunck.diagonal_image_components import get_img_diags

def _plot_help(I_xx,I_xy,I_yy,I_xt,I_yt,label):
    fig,axes = plt.subplots(1,5)
    axes[0].imshow(I_xx)
    axes[0].set_title(label+": I_xx")

    axes[1].imshow(I_xy)
    axes[1].set_title(label + ": I_xy")

    axes[2].imshow(I_yy)
    axes[2].set_title(label + ": I_yy")

    axes[3].imshow(I_xt)
    axes[3].set_title(label + ": I_xt")

    axes[4].imshow(I_yt)
    axes[4].set_title(label + ": I_yt")

def reshape_Is(Is,height,width):
    new_Is = []
    for I in Is:
        new_Is.append(I.reshape(height,width))
    return new_Is

def color_vs_gray_compare(img1_gray,img2_gray,img1_color,img2_color):
    height = img1_gray.shape[1]
    width = img1_gray.shape[2]
    gray_Is = get_img_diags(img1_gray,img1_gray)
    color_Is = get_img_diags(img1_color,img2_color)

    gray_Is = reshape_Is(gray_Is,height,width)
    color_Is = reshape_Is(color_Is,height,width)

    _plot_help(*gray_Is,label="Gray")
    _plot_help(*color_Is,label="Color")

    diff = []
    for I_color,I_gray in zip(color_Is,gray_Is):
        diff.append((I_color-I_gray)**2)
    _plot_help(*diff, label="Diff abs")

    plt.show()

def _to_gray(img):
    return np.array([(img[0]+img[1]+img[2])],dtype=float)/3

if __name__ == '__main__':
    img1_color = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10.png")
    img2_color = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11.png")
    img1_gray = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame10-gray.png")
    img2_gray = open_image(r"..\..\..\resources\eval-twoframes\Dimetrodon\frame11-gray.png")

    img1_color = _to_gray(img1_color)
    img2_color = _to_gray(img2_color)

    show_image(img1_color)
    plt.figure()
    show_image(img1_gray)
    plt.figure()
    show_image(img1_gray-img1_color)
    plt.title("Diff Image")
    plt.figure()
    plt.show()

    color_vs_gray_compare(img1_color,img2_color,img1_gray,img2_gray)
