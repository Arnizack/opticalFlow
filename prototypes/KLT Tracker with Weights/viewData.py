import matplotlib.pyplot as plt
import numpy as np
from Utilities.images import showImage,loadImage

import cv2
import numpy as np

xImgPath = "viewData/FlowX2.cvs"
yImgPath = "viewData/FlowY2.cvs" 

flowX = np.loadtxt(xImgPath)
flowY = np.loadtxt(yImgPath)

flowX = flowX
flowY = flowY

sub1 = plt.subplot(1,2,1)
sub2 = plt.subplot(1,2,2)

# Use Hue, Saturation, Value colour model 
hsv = np.zeros(shape=(flowX.shape[0],flowX.shape[1],3), dtype=np.uint8)
hsv[..., 1] = 255

mag, ang = cv2.cartToPolar(flowX, flowY)
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


showImage(sub1,bgr*6)

flowX = flowX.T
flowY = flowY.T
x = np.arange(flowX.shape[1])
y = np.arange(flowY.shape[0])

X,Y = np.meshgrid(x,y)

imgPath = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Grove\frame07Small.png"

print(flowX[16,89],flowY[16,89])

img=loadImage(imgPath)
showImage(sub2,img)

sub2.quiver(X,Y,flowX,flowY)
plt.show()

