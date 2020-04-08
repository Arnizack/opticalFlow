import numpy as np
import scipy
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import logging
from Utilities.images import loadImage
from InterpolateImg import interpolate2dArray
from ImageDerivatives import grayScaleGradient

"""
    I,T sind numpy arrays der Form (color Dimension) des Typen interpolte2d
    gradT ist ein numpy array der Form (color Dimension) des Typen interpolte2d
    N_0L und N_0U sollen einen Viereck aufspannen. Wobei N_0L der untere (lower) Punkt ist,
    und N_0U der Obere (upper)
    epislon Wert ab dem die Schrittweite klein genug ist um abgebrochen zu werden
    maxIteration ist die maximale Anzahl der Iteration die der Algorithmus machen darf
    
    die Interpolate2d Klasse sollte so Funktionieren,
    dass f([1,2,3,4],[1,2,3]) die pixel an den Punken:
    [
    [[1,1],[1,2],[1,3]]
    [[2,1],[2,2],[2,3]]
    [[3,1],[3,2],[3,3]]
    [[4,1],[4,2],[4,3]]
    ]
    zurück gibt   

    """


def setupImageForTracking(templateImgPath, targetImgPath ):
    """
    templateImgPath und targetImgPath sind Strings
    
    Zur Rückgabe:

    I,T sind numpy arrays der Form (color Dimension) des Typen interpolte2d
    die Interpolate2d Klasse gibt f([1,2,3,4],[1,2,3]) die pixel an den Punken:
    [
    [[1,1],[1,2],[1,3]]
    [[2,1],[2,2],[2,3]]
    [[3,1],[3,2],[3,3]]
    [[4,1],[4,2],[4,3]]
    ]
    zurück gibt 

    return: I,T,gradT
    """
    templImg = loadImage(templateImgPath)
    targImg = loadImage(targetImgPath)
    colorDim = templImg.shape[2]

    I = np.empty(shape=(colorDim),dtype="O")
    T = np.empty(shape=(colorDim),dtype="O")
    gradI = np.empty(shape=(colorDim,2),dtype="O")

    for c in range(colorDim):
        I[c] = interpolate2dArray(targImg[:,:,c])
        T[c] = interpolate2dArray(templImg[:,:,c])
        gradX1 = grayScaleGradient(targImg[:,:,c],"X")
        gradX2 = grayScaleGradient(targImg[:,:,c],"Y")
        gradI[c][0]=interpolate2dArray(gradX1)
        gradI[c][1]=interpolate2dArray(gradX2)
    return (I,T,gradI)


def setupTrackingRegion(xCenter,size):
    N_0L = [xCenter[0]-size,xCenter[1]-size]
    N_0U = [xCenter[0]+size,xCenter[1]+size]
    return N_0L,N_0U