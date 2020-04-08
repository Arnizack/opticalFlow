import numpy as np
import logging
from loggingHelper import *

logging.basicConfig(level=logging.DEBUG)

line = "---------------------------------"

def wolfelinesearch(p,x_0,fhandle,L,U):
    logFunction("WolfeLinesearch")
    logArgs(["p","x_0","fhandle","L","U"],[p,x_0,fhandle,L,U])
    
    
    a = 1
    p=recomputeDirection(x_0,p,L,U) #Besser wäre es das schon im BFGS zu brechnen
    if(p[0]==0 and p[1]==0):
        logging.info("Richtung ist {}".format(p))
        logReturn("Wolfelinesearch",x_0)
        return x_0
    
    x_plus,a = expandX(x_0,p,1,L,U) #Erweitern mit Schrittlänge 1
    
    
    fx_0 = fhandle.getValue(x_0)
    gradfx_0 = fhandle.getGradient(x_0)

    x_r = None
    x_l = x_0

    conditions = CeckForConditions(x_plus,fhandle,x_0,fx_0,gradfx_0,p) 
    if(conditions["ICondition"] and conditions["IICondition"]):
        logReturn("Wolfelinesearch",x_plus)
        return x_plus
    
    logging.debug("\nErweitern:")
    while(conditions["ICondition"] and not conditions["IICondition"]):
        logArgs(["ICondition","IICondition","a","x_plus","x_l","x_r"],
        [conditions["ICondition"],conditions["IICondition"],a,x_plus,x_l,x_r])
        if(checkXonBounds(x_plus,L,U)):
            #dann kann man
            logging.debug("x ist am Rand")
            logReturn("Wolfelinesearch",x_plus) 
            return x_plus
        x_l = x_plus
        x_plus,a=expandX(x_0,p,a*2,L,U,a_minus=a)
        conditions = CeckForConditions(x_plus,fhandle,x_0,fx_0,gradfx_0,p) 
    
    if(conditions["ICondition"] and conditions["IICondition"]):
        logReturn("Wolfelinesearch",x_plus)
        return x_plus
    logging.debug("\n")
    logging.debug("Intervall Halbierung")

    #Wenn jetzt noch da, gilt I nicht
    x_r = x_plus
    while(not conditions["ICondition"] or not conditions["IICondition"]):
        logArgs(["ICondition","IICondition","x_r","x_l","x_plus","fx_plus","gradfx_plus"],
        [conditions["ICondition"],conditions["IICondition"],x_r,x_l,x_plus,fhandle.getValue(x_plus),
        fhandle.getGradient(x_plus)])
        if(not conditions["ICondition"]):
            #Zulang
            x_r=x_plus
        elif(not conditions["IICondition"]):
            x_l=x_plus

        x_plus = (x_r+x_l)/2
        x_plus=x_plus.astype(np.int32)
        if((x_plus==x_r).min() or (x_plus==x_l).min()):
            return x_plus
        conditions = CeckForConditions(x_plus,fhandle,x_0,fx_0,gradfx_0,p) 
    
    logReturn("Wolfelinesearch",x_plus)    
    return x_plus
    
        
def expandX(x_0,p,a,L,U,a_minus=0):
    """
    x_0 Ausgangscoordinate
    p Richtung
    a Schrittweite
    L Lower Bound
    U Upper Bound
    a_minus die letzte Schrittweite

    Annahme x_0 in Bounds und letzter Schritt mindestens einen Pixel von der Grenze entfernt
    """
    if(not checkXinBounds(x_0+p*a_minus,L,U)):
        raise Exception("der Letzte Schritt ist nicht im Box Constraint")
    x = x_0+a*p
    x=x.astype(np.int32)
    while(not checkXinBounds(x,L,U)):
        a=(a+a_minus)/2
        x = x_0+a*p
        x=x.astype(np.int32)

        if((x==x_0).min()):
            raise Exception("dann x_0 an der Grenze das darf nicht sein")
    
    return (x,a)
    
def recomputeDirection(x_0,p,L,U):
    #Testen ob x_0 auf der Grenze Liegt
    logFunction("recomputeDirection")
    logArgs(["x_0","p","L","U"],[x_0,p,L,U])
    for i in range(len(x_0)):
        if(x_0[i]==L[i] and p[i]<0):
            #Also falls x auf der unteren Grenze ist und die Richtung nach unten zeigt
            p[i]=0
        
        if(x_0[i]==U[i] and p[i]>0):
            #Also falls x auf der oberen Grenze ist und die Richtung nach oben zeigt
            p[i]=0
    if(abs(p[0])>=abs(p[1]) and abs(p[0])<1):
        p/=abs(p[0])
    elif(abs(p[1])>=abs(p[0]) and abs(p[1])<1):
        p/=abs(p[1])
    
    logReturn("recomputeDirection",p)
    return p

def checkXonBounds(x,L,U):
    for i in range(len(x)):
        if(x[i]==L[i] or x[i]==U[i]):
            return True
    return False
        
        
def checkXinBounds(x,L,U):
    for i in range(len(x)):
        if(x[i]<L[i]):
            return False
        if(x[i]>U[i]):
            return False
    return True


def CeckForConditions(x,fhandle,x_0,fx_0,gradfx_0,p):
    alpha = 0.01
    beta = 0.9

    ICondition= False
    fx = fhandle.getValue(x)
    gradfx = fhandle.getGradient(x)
    deltax= x-x_0
    ICondition = fx<= alpha*gradfx_0.dot(deltax)+fx_0
    IICondition = gradfx.dot(p)>=beta*gradfx_0.dot(p)
    

    return {"ICondition":ICondition, "IICondition":IICondition}





