import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from InterpolateImg import Interpolator
from Utilities.loggingHelper import *
import logging

def computeH(gradx1,gradx2):
    gradx1 = gradx1.flatten()
    gradx2 = gradx2.flatten()
    
    H = np.array([[gradx1**2,gradx2*gradx1],
                  [gradx2*gradx1,gradx2**2]])
    
    H = H.sum(axis=2)
    return H

def computeZ(I,T,gradTX1,gradTX2):
    #Schritt 1
    e = I-T
    #Schritt 2
    gradX1 = gradTX1*e
    gradX2 = gradTX2*e
    
    return np.array([gradX1.sum(),gradX2.sum()])    

def trackTamplate(I,T,gradT,N_0L,N_0U,epsilon,maxIteration):
    """
    I,T sind numpy arrays der Form (color Dimension) des Typen interpolte2d
    gradT ist ein numpy array der Form (color Dimension) des Typen interpolte2d
    N_0L und N_0U sollen einen Viereck aufspannen. Wobei N_0L der untere (lower) Punkt ist,
    und N_0U der Obere (upper)
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
    
    #Sicherheit
    if(len(I)!=len(T) or len(I)!=len(gradT)):
        raise Exception("I,T,gradT haben unterschiedliche Farb Dimensionen")
    if(gradT.shape[1]!=2):
        raise Exception("gradT muss 2D Gradienten haben, also folgende shape (colorDimension,2)")

    for interpolateObjecte in zip(I,T,gradT[:,0],gradT[:,1]):
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


    #Schritt 1:
    colorRange = len(I)
    p = np.array([0,0])
    deltap = np.array([100,0])
    #Man braucht nicht immer durch x durchloop aufgrund von Numpy

    x1Range = np.arange(N_0L[0],N_0U[0])    
    x2Range = np.arange(N_0L[1],N_0U[1])

    H = np.array([[0,0],[0,0]],dtype=np.float64)
    T_x = np.empty(shape =(colorRange,len(x1Range),len(x2Range)))
    gradT_x = np.empty(shape =(colorRange,2,len(x1Range),len(x2Range)))

    iterator = 0
    
    
    #Schritt 2
    
    for c in range(colorRange):
        T_x[c] = T[c](x1Range,x2Range)
        #Schritt 2.A
        gradTx1 = gradT[c][0](x1Range,x2Range)
        gradTx2 = gradT[c][1](x1Range,x2Range)
        gradT_x[c] = [gradTx1,gradTx2]
        #Schritt 2.B
        H_c = computeH(gradTx1,gradTx2)
        logArgs(["H","H_c"],[H,H_c])
        #Schritt 2.C
        H += H_c
    
    #Schritt 3
    Hinv = np.linalg.inv(H)
    logArgs(["H","Hinv"],[H,Hinv])
    
    
    while(np.linalg.norm(deltap)>epsilon and iterator<maxIteration):
    
        #Schritt 4:
        x1tildeRange = x1Range+p[0]
        x2tildeRange = x2Range+p[1]
        
        #Schritt 5:
        Z=np.array([0,0],dtype=np.float64)
        
        #Schritt 6:
        for c in range(colorRange):
            #Schritt A & B
            Ixtilde_c = I[c](x1tildeRange,x2tildeRange)
            T_c = T_x[c]
            gradTx1_c = gradT_x[c][0]
            gradTx2_c = gradT_x[c][1]
            Z_c = computeZ(Ixtilde_c,T_c,gradTx1_c,gradTx2_c)
            #Schritt C
            Z += Z_c
        
        #Schritt 7:
        deltap = -Hinv.dot(Z)
        
        logArgs(["Hinv","Z","deltap","p","Iterations"],[Hinv,Z,deltap,p,iterator])

        #Schritt 8:
        p = p + deltap
        
        #Iterator erhöhen
        iterator += 1
        
    return p 