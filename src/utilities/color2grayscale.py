import numpy as np
from src.preprocessing.rof import denoising_chambolle
import matplotlib.pyplot as plt

def color2grayscale(image : np.ndarray, methode="heuristic_linear_combination") -> np.ndarray:
    """
    :param image: (3,height,width)
    :param methode: "heuristic_linear_combination" ,"ROF"
    :return: (height,width)
    """
    if(image.shape[0]==1):
        return image[0]

    R = image[0]
    G = image[1]
    B = image[2]

    gray_img = 0.2126 * R + 0.7152 * G + 0.0722 * B

    if (methode == "heuristic_linear_combination"):
            return gray_img

    if(methode=="ROF"):
        std = gray_img.std()
        structured, textured = denoising_chambolle(gray_img, lambda0=0.125, std_dev=0.5, iter=1)
        #mix = 0.9
        #*(1-mix) + structured *mix
        result = textured*4/5 + structured*1/5

        #result = (result-result.min())/(result.max()-result.min())
        print("ROF Result Min: ",result.min())
        print("ROF Result Max: ",result.max())
        plt.imshow(result)
        plt.figure()
        plt.imshow(gray_img)
        plt.show()
        return result