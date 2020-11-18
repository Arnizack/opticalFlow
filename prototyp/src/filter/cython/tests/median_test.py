import os
import sys

sys.path.append(os.path.realpath("."))

from bilateral_median import quickselect_median
import numpy as np

if __name__ == "__main__":
    a = np.array([3,4,2,1,5,6],dtype=np.float32)
    median=quickselect_median(a)
    print("Median: ",median)