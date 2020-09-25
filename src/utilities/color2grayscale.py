import numpy as np
from src.preprocessing.rof import denoising_chambolle

def color2grayscale(image : np.ndarray, methode="heuristic_linear_combination") -> np.ndarray:
    """
    :param image: (3,height,width)
    :param methode: "heuristic_linear_combination" ,"rof"
    :return: (height,width)
    """


    R = image[0]
    G = image[1]
    B = image[2]

    gray_img = 0.2126 * R + 0.7152 * G + 0.0722 * B

    if (methode == "heuristic_linear_combination"):
            return gray_img

    if(methode=="rof"):
        structured, textured = denoising_chambolle(gray_img, lambda0=0.125, std_dev=std, iter=20)
        return textured
