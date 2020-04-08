import math
def map2DVectorToColor(vec):
    length = (vec[0]**2 + vec[1]**2)**0.5
    rotation = math.acos(vec[0]/length)

    return [0,length,rotation+1]