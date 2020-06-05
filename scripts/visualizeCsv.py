import sys
import flow_vis
import numpy as np
import matplotlib.pyplot as plt
#filepath = sys.argv[1]

filepath = r"H:\dev\opticalFlow\src\out\build\x64-Release\intern\cpu\cpuoptflow\tracking.csv"

result = None
width = 0
heigth = 0

with open(filepath,"r") as file:
    line1 = file.readline()
    width,heigth = line1.split(";")
    width,heigth = int(width), int(heigth)
    result = np.loadtxt(file,delimiter=";")

result.shape=(heigth,width,2)

result = np.flip(result,0)

flow_color = flow_vis.flow_to_color(result, convert_to_bgr=False)

plt.imshow(flow_color)
plt.show()