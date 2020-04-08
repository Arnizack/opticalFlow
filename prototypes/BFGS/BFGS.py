import numpy as np
import logging
from loggingHelper import *
from wolfeLinesearch import wolfelinesearch, recomputeDirection

def updateInversHessian(H,y,s):
    """
    Alles numpy arrays
    """
    logFunction("updateInversHession",logLevel=15)
    logArgs(["y","s"],[y,s],logLevel=15)

    sTy = s.dot(y)
    yTHy = y.dot(H).dot(y)
    ssT = np.array([s[0]*s,s[1]*s])

    V=(sTy+yTHy)*(ssT)/(sTy**2)

    Hy = H.dot(y)
    HysT = np.array([Hy[0]*s,Hy[1]*s])
    
    syT = np.array([s[0]*y,s[1]*y])
    syTH = syT.dot(H)

    U=(HysT+syTH)/(sTy)
    
    H_plus = H+V-U

    logReturn("updateInversHessian",H_plus,logLevel=15)
    return H_plus

def computeH0(y,s):
    yTs = y.dot(s)
    yTy = y.dot(y)

    H = yTs/yTy*np.array([[1,0],[0,1]])
    return H

class BFGS:
    terminationThreashold = 0.01
    descentThreashold = 0.9
    gradientEncreaseThreashold = 0.1
    maxIterations = 1000
    _counter = 0

    def __init__(self):
        self._counter = 0

    def BFGSminimize(self,x_0,fhandle,L,U):
        a=0.01
        logFunction("BFGSminimize",logLevel=15)
        logArgs(["x_0","fhandle","L","U"],[x_0,fhandle,L,U])
        x= x_0
        gradfx = fhandle.getGradient(x)
        p = -gradfx
        p = recomputeDirection(x_0,p,L,U)
        
        firstRun=True

        H=None
        x_minus = L-np.array([1,1])

        while(not self.checkConvergence(p,x,gradfx,x_minus)):
            logLine(logLevel=15)
            logging.log(15,"New Iteration")

            x_plus = wolfelinesearch(p,x,fhandle,L,U)
            gradfx_plus = fhandle.getGradient(x_plus)
            if(x_plus[0]==x[0] and x_plus[1]==x[1]):
                logging.log(15,"Repeatable X")
                break
            logging.log(15,"x: {}\tgradfx: {}".format(x_plus,gradfx_plus))

            y = gradfx-gradfx_plus
            s = x-x_plus
            
            if(firstRun):
                firstRun = False
                H = computeH0(y,s) 
                logging.log(15,"temporary frist H: {}".format(H))


            H = updateInversHessian(H,y,s)
            logging.log(15,"H: {}".format(H))

            x_minus = x
            x = x_plus
            gradfx = gradfx_plus
            p = -H.dot(gradfx)
            p = recomputeDirection(x,p,L,U)

        logReturn("BFGSminimize",x)
        return x

    def checkConvergence(self,p,x,gradfx,x_minus):
        #p sollte bereits vorberechnet sein und auf L U angepasst sein

        #Test Gradient
        gradientTest = (gradfx[0]**2+gradfx[1]**2)<self.terminationThreashold

        #DirectionTest
        directionTest = (p[0]==0 and p[1]==0)

        #IterationTest
        iterationTest = self._counter>self.maxIterations

        #repeatableTest
        repeatableTest = x_minus[0]==x[0] and x_minus[1]==x[1]

        logging.log(15,
        "gradientTest: {} | directionTest:{} | titerationTest: {} | repeatableTest: {} ".format(gradientTest,directionTest,iterationTest,repeatableTest))

        result = gradientTest or directionTest or iterationTest or repeatableTest

        logging.log(15,"ConvergenceTests: {}".format(result))
        logging.log(15,"Iteration: {}".format(self._counter))
        self._counter+=1
        return result 