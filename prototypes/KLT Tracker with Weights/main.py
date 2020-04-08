import matplotlib.pyplot as plt
from setupTracking import setupImageForTracking,setupTrackingRegion
from simpleFlow_tracker import trackTamplate
from Utilities.images import showToPointsOnTowImgs, showImage, loadImage, showGradient
import logging

logging.basicConfig(level=10)
"""
x_center = [4,82]
imgTemplate = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Grove\frame07Small.png"
imgNext = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Grove\frame08Small.png"
"""
"""
x_center = [126,183]
imgTemplate = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Wooden\frame07.png"
imgNext = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Wooden\frame08.png"
"""
"""
x_center = [318,341]
imgTemplate = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Teddy\frame10.png"
imgNext = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Teddy\frame11.png"
"""
"""
x_center = [70,223]
imgTemplate = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\GradientReferenze\frame01.png"
imgNext = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\GradientReferenze\frame02.png"
"""

x_center = [94,72]
imgTemplate = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Der See Schild\sframe07.jpg"
imgNext = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Der See Schild\sframe08.jpg"

featureSize = 5
epsilon = 0.01
maxIteration = 40
delta_c = 1
delta_d =  5.5

I,T,gradI = setupImageForTracking(imgTemplate,imgNext)
N_0L,N_0U = setupTrackingRegion(x_center,featureSize)

showGradient(plt,gradI,I[0].x_max-I[0].x_min,I[0].y_max-I[0].y_min)
plt.show()

p = trackTamplate(I,T,gradI,N_0L,N_0U,x_center,delta_c,delta_d,epsilon,maxIteration)



x_ziel = [x_center[0]+p[0],x_center[1]+p[1]]

print("X Ziel: ",x_ziel)
showToPointsOnTowImgs(plt,loadImage(imgTemplate),loadImage(imgNext),x_center,x_ziel)
plt.show()