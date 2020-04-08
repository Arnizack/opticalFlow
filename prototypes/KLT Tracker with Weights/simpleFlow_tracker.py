import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from InterpolateImg import Interpolator
from Utilities.loggingHelper import *
from Utilities.images import showImage
import logging

def computeWeights(x1,x2,T_x,delta_c,delta_d,x_0,Tx_0):
    #Schritt 1
    distanceX1 = x1-x_0[0]
    distanceX2 = x2-x_0[1]
    #Schritt2
    normDistance = distanceX1**2 + distanceX2**2
    
    normColor = None
    cDim = T_x.shape[0]
    
    for c in range(cDim):
        #Schritt 3
        colorDiff = T_x[c]-Tx_0[c]
        #Schritt 4        
        if(normColor is None):
            normColor = colorDiff**2
        else:
            normColor += colorDiff**2
    
    #Schritt 5
    exponentDistance = -1* normDistance/(2*delta_d)
    
    #Schritt 6
    exponentColor = -1*normColor/(2*delta_c)
    
    #Schritt 7
    weights = np.exp(exponentColor+exponentDistance)

    return weights

def computeH(gradx1,gradx2,weights):
    # gradx1, gradx2,weights solltem 1D sein
    #Schritt f
    H = np.array([[gradx1**2,gradx2*gradx1],
                  [gradx2*gradx1,gradx2**2]])
    #Schritt g
    H[0][0]*=weights
    H[0][1]*=weights
    H[1][0]*=weights
    H[1][1]*=weights
    
    return H.sum(axis=2)

def computeZ(I,T,gradIX1,gradIX2,weights):
    #Schritt b
    e = I-T
    #Schritt c
    gradX1 = gradIX1*e
    gradX2 = gradIX2*e
    #Schritt d
    ZX1 = weights * gradX1
    ZX2 = weights * gradX2
    
    return np.array([ZX1.sum(),ZX2.sum()])   

def trackTamplate(I,T,gradI,N_0L,N_0U,x_0,delta_c,delta_d,epsilon,maxIteration):
    """
    I,T sind numpy arrays der Form (color Dimension) des Typen interpolte2d
    gradI ist ein numpy array der Form (color Dimension) des Typen interpolte2d
    N_0L und N_0U sollen einen Viereck aufspannen. Wobei N_0L der untere (lower) Punkt ist,
    und N_0U der Obere (upper)
    x_0 ist der betrachtete Mittelpunkt und eine 2D List
    delta_c ist eine Zahl, die den Color weight bestimmt
    delta_d ist eine Zahl, die den Distance weight bestimmt
    epislon Wert ab dem die Schrittweite klein genug ist um abgebrochen zu werden
    maxIteration ist die maximale Anzahl der Iteration die der Algorithmus machen darf
    
    die Interpolate2d Klasse sollte so Funktionieren,
    dass f([1,2,3,4],[1,2,3]) die pixel an den Punken:
    [
    [[1,1],[1,2],[1,3]]
    [[2,1],[2,2],[2,3]]
    [[3,1],[3,2],[3,3]]
    [[4,1],[4,2],[4,3]]
    ]
    zurück gibt   

    """
    logFunction("SimpleFlow Tracker")
    logArgs(["I","T","gradI","N_0L","N_0U","x_0","delta_c","delta_d","epsilon","maxIteration"],[I,T,gradI,N_0L,N_0U,x_0,delta_c,delta_d,epsilon,maxIteration])

    #Sicherheit
    if(len(I)!=len(T) or len(I)!=len(gradI)):
        raise Exception("I,T,gradT haben unterschiedliche Dimensionen")
    
    if(gradI.shape[1]!=2):
        raise Exception("gradT muss 2D Gradienten haben, also folgende shape (colorDimension,2)")

    for interpolateObjecte in zip(I,T,gradI[:,0],gradI[:,1]):
        x_min = None
        x_max = None
        y_min = None
        y_max = None
        for interpolateObj in interpolateObjecte:
            if(type(interpolateObj)!= Interpolator):
                raise Exception("I oder T oder gradT enthalten kein interp2d Objekt")

            if(x_min is None or x_max is None or y_min is None or y_max is None):
                x_min = interpolateObj.x_min
                x_max = interpolateObj.x_max
                y_min = interpolateObj.y_min
                y_max = interpolateObj.y_max
            else:
                if(x_min != interpolateObj.x_min or
                x_max != interpolateObj.x_max or
                y_min != interpolateObj.y_min or
                y_max != interpolateObj.y_max):
                    raise Exception("Einer der interpolate Objekte hat eine anderen x,y Raum als die anderen")
    
    if(type(N_0L)!=list):
        raise Exception("N_0L ist keine Liste")
    if(len(N_0L)!=2):
        raise Exception("N_0L hat nicht von Dimension 2")

    if(type(N_0U)!=list):
        raise Exception("N_0U ist keine Liste")
    if(len(N_0U)!=2):
        raise Exception("N_0U hat nicht von Dimension 2")

    if(type(x_0)!=list):
        raise Exception("x_0 ist keine Liste")
    if(len(x_0)!=2):
        raise Exception("x_0 hat nicht von Dimension 2")
        


    #Schritt 1 & 2 
    cDim = len(I)
    p = np.array([0,0],dtype=np.float64)
    deltap = np.array([epsilon,epsilon])

    x1Range = np.arange(N_0L[0],N_0U[0])    
    x2Range = np.arange(N_0L[1],N_0U[1])
    x1Dim = len(x1Range)
    x2Dim = len(x2Range)
    
    iterator = 0
    
    #Schritt 3
    T_x = np.empty(shape=(cDim,x1Dim,x2Dim),dtype = np.float64)
    T_x0 = np.empty(shape=(cDim),dtype = np.float64)
    
    for c in range(cDim):
        T_x[c] = T[c](x1Range,x2Range)
        T_x0[c] = T[c](x_0[0],x_0[1])
        
    #Schritt 4
    X1,X2 = np.meshgrid(x1Range,x2Range)
    weights = computeWeights(X1,X2,T_x,delta_c,delta_d,x_0,T_x0)
    
    logging.debug("Weights:")
    
    showImage(plt,weights)
    plt.show()
    
    logging.debug("Weights Shape: {}".format(weights.shape))

    #Schritt 5
    while(np.linalg.norm(deltap)>epsilon and iterator<maxIteration):
        
        #Schritt A
        H = np.array([[0,0],[0,0]],np.float64)
        
        #Schritt B
        Z = np.array([0,0],np.float64)
        
        #Schritt C
        x1Tilde = x1Range+p[0]
        x2Tilde = x2Range+p[1]
        
        #Schritt D
        for c in range(cDim):
            #Schritt a
            gradIX1 = gradI[c][0](x1Tilde,x2Tilde)
            gradIX2 = gradI[c][1](x1Tilde,x2Tilde)
            
            #Schritt b - d
            I_x = I[c](x1Tilde,x2Tilde)
            Z_c = computeZ(I_x,T_x[c],gradIX1,gradIX2,weights)
            
            #Schritt e
            Z+=Z_c
            
            #Schritt f-g
            H_c = computeH(gradIX1.flatten(),gradIX2.flatten(),weights.flatten())
            
            #Schritt h
            H+=H_c
            
        #Schritt E
        Hinv = np.linalg.inv(H)
        

        #Schritt F
        deltap = -Hinv.dot(Z)
        
        #Schritt G
        p +=deltap
        logArgs(["Hinv","Z","deltap","p","Iterations"],[Hinv,Z,deltap,p,iterator])
        
        iterator+=1

    return p
            
        
    