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
    

    logHeader="{:"+str(argsMaxLength)+"}|"
    logHeader*=len(argsName)
    
    
    strArgs=[]
    strNameArgs=[]
    for arg in args:
        hinzu =str(arg)
        if(len(hinzu)>argsMaxLength):
        
            arg = hinzu[0:argsMaxLength]
        
        strArgs.append(hinzu)

    for arg in argsName:
        if(len(str(arg))>argsMaxLength):
        
            arg = str(arg)[0:argsMaxLength]
        
        strNameArgs.append(str(arg))

    logHeaderMsg=logHeader.format(*strNameArgs)
    logArgsMsg = logHeader.format(*strArgs)
    logging.log(logLevel,logHeaderMsg)
    logging.log(logLevel,logArgsMsg)

def logLine(logLevel=10,zeihen="-",laenge = 100):
    logging.log(logLevel,zeihen*laenge)

def logFunction(funcName,logLevel = 10):
    logLine(logLevel=logLevel)
    logging.log(logLevel,funcName)
    logLine(logLevel=logLevel)

def logReturn(funcName,value,logLevel=10):
    logging.log(logLevel,"{} return: {}".format(funcName,value))
