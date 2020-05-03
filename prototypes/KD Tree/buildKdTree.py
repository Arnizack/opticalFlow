import numpy as np


from Utilities.images import loadImage



class Node:
    dimension = 0
    cut = 0
    left = None
    right = None
    nmax = 0
    nmin = 0



if __name__ == "__main__":
    path = r"H:\OneDrive\Projekte\SimpleFlow\eval-data\Evergreen\frame14.png"
    img = loadImage(path)
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    X,Y = np.meshgrid(x,y)
    X = X.reshape((img.shape[0]*img.shape[1]))
    Y = Y.reshape((img.shape[0]*img.shape[1]))
    img=img.reshape((img.shape[0]*img.shape[1],img.shape[2]))
    img*=1000
    img = img.astype(np.int)

    global samples
    samples = np.empty(shape = img.shape[0],dtype= np.int64)

    while(True):

        samples+= (img[:,0]%2)*10
        img[:,0] = (img[:,0]/2).astype(np.int)

        samples+= (img[:,1]%2)*10
        img[:,1] = (img[:,1]/2).astype(np.int)

        samples+= (img[:,2]%2)*10
        img[:,2] = (img[:,2]/2).astype(np.int)


 

        samples+= (X%2)*10
        X = (X/2).astype(np.int)
        
        samples+= (Y%2)*10
        Y = (Y/2).astype(np.int)
        

        if(img[0].sum()==0 and img[1].sum()==0  and img[2].sum()==0 and X.sum() == 0 and Y.sum() == 0):
            break
    
    print(samples)
    