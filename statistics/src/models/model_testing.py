import numpy as np
import sun_baker_model
import matplotlib.pyplot as plt

class ModelTestResult:
    def __init__(self,name,error_map):
        self.error_map = error_map
        self.name = name
    
    def evaluate(self,methode="mean"):
        if(methode=="median"):
            return np.median(self.error_map)
        
        return np.mean(self.error_map)
    
    def plot(self):
        print(self.name)
        plt.imshow(self.error_map)
        print("Median: ",self.evaluate("median"),"\tMean: ",self.evaluate("mean"))
        

    