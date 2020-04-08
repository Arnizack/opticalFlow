import numpy as np
import scipy
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import logging
from Utilities.loggingHelper import *

def _translateDirection1(np_img):
    np_temp=np.swapaxes(np_img,0,1)
    np_result=np.flip(np_temp,axis=1)
    return np_result

def _translateDirection2(np_img):
    
    np_temp=np.flip(np_img,axis=1)
    np_result=np.swapaxes(np_temp,0,1)
    return np_result


def loadImage(path):
    """
    importiert ein Bild
    BSP. Koordinatenraum des Bildes:
    1|2
    3|4

    3 hat die Koordinate x:0,y:0
    4 hat die Koordinate x:1,y:0
    1 hat die Koordinate x:0,y:1
    2 hat die Koordinate x:1,y:1    
    Sei img das zurückgegebene Bild
    Abfrage: img[x][y]
    """

    np_img = imread(path)
    
    #Koorigieren des Koordinatenraums
    np_result = _translateDirection1(np_img)       

    if(np_result.shape[2]>3):
        np_result = np_result[:,:,0:3]
    return np_result

def convertImageToGrayScale(np_img):
    np_result = np.empty(np_img.shape[0:2])
    for x in range(np_img.shape[0]):
        for y in range(np_img.shape[1]):
            np_result[x][y]=np_img[x][y].sum()/np_img.shape[2]
    return np_result

def showImage(ax,np_img,colormap="cividis"):
    np_converted = _translateDirection2(np_img)
    shape = np_converted.shape
    if(len(shape)==3 and shape[2]==1):
        np_converted=np_converted[:,:,0]
    ax.imshow(np_converted,extent=[0,shape[1],0,shape[0]],cmap=colormap)

def saveCSV(np_img,name):
    path=name+".cvs"
    np.savetxt(path,np_img)
    
def loadCSV(path):
    return np.loadtxt(path)

def shapeCXYToXYC(img):
    xdim = img.shape[1]
    ydim = img.shape[2]
    cdim = img.shape[0]
    newImg = np.empty(shape=(xdim,ydim,cdim))
    for c in range(cdim):
        newImg[:,:,c] = img[c]
    
    #img.shape = (xdim,ydim,cdim)
    return newImg

def showGradient(ax,gradient,xdim,ydim,subplotStartX=0,subplotStartY=0):
    colorRange = len(gradient)
    subplots = []
    if(type(gradient[0][0])==np.ndarray):
        gradX = gradient[:,0]
        gradY = gradient[:,1]
    else:
        x = np.arange(xdim)
        y = np.arange(ydim)
        gradX = np.empty(shape=(colorRange,xdim,ydim))
        gradY = np.empty(shape=(colorRange,xdim,ydim))
        for colorIdx in range(colorRange):
            gradX[colorIdx]=gradient[colorIdx][0](x,y)
            gradY[colorIdx]=gradient[colorIdx][1](x,y)
    gradXMin = gradX.min()
    gradYMin = gradY.min()
    gradXMax = gradX.max()
    gradYMax = gradY.max()
    gradXNull = (0-gradXMin)/(gradXMax-gradXMin)
    gradYNull = (0-gradYMin)/(gradYMax-gradYMin)

    logArgs(["gradXMax","gradXNull","gradXMin","gradYMax","gradYNull","gradYMin"],
    [gradXMax,gradXNull,gradXMin,gradYMax,gradYNull,gradYMin])
    gradX=(gradX-gradXMin)/(gradXMax-gradXMin)
    gradY=(gradY-gradYMin)/(gradYMax-gradYMin)
    gradX

    subax1 = ax.subplot(subplotStartX+1,subplotStartY+2,subplotStartX*subplotStartY+1)
    subax2 = ax.subplot(subplotStartX+1,subplotStartY+2,subplotStartX*subplotStartY+2)
    conGradX = shapeCXYToXYC(gradX)
    conGradY = shapeCXYToXYC(gradY)
    showImage(subax1,conGradX)
    showImage(subax2,conGradY)
    return (subax1,subax2)

def showToPointsOnTowImgs(ax,img1,img2,point1,point2,colormap="cividis"):
    sub1 = ax.subplot(1,2,1)
    sub2 = ax.subplot(1,2,2)
    showImage(sub1,img1,colormap=colormap)
    showImage(sub2,img2,colormap=colormap)
    sub1.scatter(point1[0],point1[1])
    sub2.scatter(point2[0],point2[1])

if __name__ == "__main__":
    from scipy import interpolate
    path = r"H:\OneDrive\Projekte\SimpleFlow\KLT Tracker\testData\CalibrationPicture2.png"
    path2 = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\GradientReferenze\frame01.png"
    img = loadImage(path2)
    print(img)



    showImage(plt,img,colormap="gray")
    plt.show()