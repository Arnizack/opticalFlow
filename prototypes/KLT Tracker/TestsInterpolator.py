from InterpolateImg import interpolate2dArray
import numpy as np

def test_helper(feld):
    feld = np.array(feld)
    interpObj = interpolate2dArray(feld)

    actualInterpolate = interpObj(np.arange(feld.shape[0]),np.arange(feld.shape[1]))
    if(actualInterpolate.shape != feld.shape):
        print("interpolateObj hat andere Shape als feld")
        return False

    for row in zip(actualInterpolate,feld):
        for actualItem,expectedItem in zip(*row):
            
            if(actualItem!=expectedItem):
                print("Die Items stimmen nicht")
                return False
    return True


if __name__ == "__main__":
    feld = np.arange(5*7)
    feld.shape=(5,7)
    if(test_helper(feld)):
        print("Test war erfolgreich")
    else:
        print("Test hat fehlgeschlagen")
    


    feld = np.random.rand(3,2)
    if(test_helper(feld)):
        print("Test war erfolgreich")
    else:
        print("Test hat fehlgeschlagen")
