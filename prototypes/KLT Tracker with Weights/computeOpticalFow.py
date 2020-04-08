import matplotlib.pyplot as plt
from setupTracking import setupImageForTracking,setupTrackingRegion
from simpleFlow_tracker import trackTamplate
from Utilities.images import showToPointsOnTowImgs, showImage, loadImage, showGradient, saveCSV
from Utilities.vectorShow import map2DVectorToColor
import logging
import numpy as np

if __name__ == "__main__":
        
    logging.basicConfig(level=40)
    """
    imgTemplate = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Grove\frame07Small.png"
    imgNext = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Grove\frame08Small.png"
    """
    imgTemplate = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Army\frameSmall07.png"
    imgNext = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Army\frameSmall08.png"

    featureSize = 5
    epsilon = 0.01
    maxIteration = 40
    delta_c = 0.5
    delta_d =  2

    I,T,gradI = setupImageForTracking(imgTemplate,imgNext)
    xDim = I[0].x_max-I[0].x_min
    yDim = I[0].y_max-I[0].y_min

    result = np.zeros((xDim,yDim,2),dtype=np.float64)

    for x in range(featureSize,xDim-featureSize):
        for y in range(featureSize,yDim-featureSize):
            x_center = [x,y]
            N_0L,N_0U = setupTrackingRegion(x_center,featureSize)
            try:
                p = trackTamplate(I,T,gradI,N_0L,N_0U,x_center,delta_c,delta_d,epsilon,maxIteration)
                #color = map2DVectorToColor(p)
            except:
                p=[0.1,0.1]
            
            result[x][y]=p
            print(x,y,p)
    
    saveCSV(result[:,:,0],"FlowX")
    saveCSV(result[:,:,1],"FlowY")