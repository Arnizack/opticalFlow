import matplotlib.pyplot as plt
import glob
from flowiz import  flowiz
import numpy as np
import matplotlib.pyplot as plt

def _reshape_flow(field,width,height):
    reshape_flow = np.reshape(field, [2, height, width])

    return np.array([reshape_flow[1],reshape_flow[0]])

def show_flow_field(field,width,height, axes = plt,mode="RGB"):
    """
    Displays a Flow Field
    mode = "RGB"
    :return: None
    """
    resharped_flied = _reshape_flow(field,width,height)
    flowiz_field = np.empty(shape=(height,width,2),dtype=float)
    flowiz_field[:,:,0]=resharped_flied[0]
    flowiz_field[:, :, 1] = resharped_flied[1]
    image=flowiz.convert_from_flow(flowiz_field,mode="RGB")
    axes.imshow(image)

def show_flow_field_arrow(field,width,height, axes = plt):
    X,Y =np.meshgrid(np.arange(width),np.arange(height))
    resharped_flied = _reshape_flow(field, width, height)
    coords = [X.flatten(),Y.flatten()]
    directions = [resharped_flied[0].flatten(),resharped_flied[1].flatten()]
    axes.quiver(*coords,*directions)


def show_flow_difference(field1,field2,width,height):
    #crop Images
    diff_field = field1[:,0:width,0:height]-field2[:,0:width,0:height]
    show_flow_field(diff_field,width,height)


def read_flow_field(path):
    file = glob.glob(path)
    in_flowfield = flowiz.read_flow(path)
    #reformat
    converted_flowfield = np.array([in_flowfield[:,:,1],in_flowfield[:,:,0]])
    return converted_flowfield

