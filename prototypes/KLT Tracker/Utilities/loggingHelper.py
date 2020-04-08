import logging

def logArgs(argsName,args,logLevel=10,argsMaxLength=20):
    """
    argsName ist ein Array
    args ist ein Array
    """
    if(len(argsName)!=len(args)):
        print("Fehler beim Logen, Anzahl der Args und deren Namen stimmt nicht")
        print("ArgsNames: ",argsName)
        print("Args: ",args)
    

    logHeaderSingle="{:"+str(argsMaxLength)+"}|"
    logHeader=logHeaderSingle*len(argsName)
    
    
    strArgs=[]
    strNameArgs=[]
    maxHeight = 0
    for arg in args:
        argStr =str(arg)
        argStrLines = argStr.splitlines()

        
        for idx,argStrLine in enumerate(argStrLines):
            if(len(argStrLine)>argsMaxLength):
            
                argStrLines[idx] = argStrLine[0:argsMaxLength]
        if(len(argStrLines)>maxHeight):
            maxHeight = len(argStrLines)

        strArgs.append(argStrLines)

    for arg in argsName:
        hinzu =str(arg)
        if(len(hinzu)>argsMaxLength):
        
            hinzu = str(arg)[0:argsMaxLength]
        
        strNameArgs.append(hinzu)

    logHeaderMsg=logHeader.format(*strNameArgs)
    logging.log(logLevel,logHeaderMsg)
    for idxLine in range(maxHeight):
        logMsgs = []
        for strArgLines in strArgs:
            if(len(strArgLines)>idxLine):
                logMsgs.append(strArgLines[idxLine])
            else:
                logMsgs.append("")
        logArgsMsg = logHeader.format(*logMsgs)
        logging.log(logLevel,logArgsMsg)

    #logging.log(logLevel,logArgsMsg)

def logLine(logLevel=10,zeihen="-",laenge = 100):
    logging.log(logLevel,zeihen*laenge)

def logFunction(funcName,logLevel = 10):
    logLine(logLevel=logLevel)
    logging.log(logLevel,funcName)
    logLine(logLevel=logLevel)

def logReturn(funcName,value,logLevel=10):
    logging.log(logLevel,"{} return: {}".format(funcName,value))


if __name__ == "__main__":
    logging.basicConfig(level=10)
    import numpy as np
    feld = np.array([[1,2],[3,4]])
    logArgs(["Str","Feld"],["Hallo",feld])