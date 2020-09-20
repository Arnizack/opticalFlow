import numpy as np


def color2grayscale(image : np.ndarray, methode="heuristic_linear_combination") -> np.ndarray:
    """
    :param image: (3,height,width)
    :param methode: string
    :return: (height,width)
    """

    if(methode == "heuristic_linear_combination"):
        R = image[0]
        G = image[1]
        B = image[2]
        return 0.2126 * R + 0.7152 * G + 0.0722 * B
