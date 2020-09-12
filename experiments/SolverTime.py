from scipy import sparse
import inspect
from scipy.sparse import linalg
from scipy.sparse.linalg import dsolve

import  matplotlib.pyplot as plt

import numpy as np

from time import time


width = 1920
height = 1080

count = int(width*height*2)

randGen = np.random.default_rng()


filterEcken = np.full(fill_value=3/6,shape=(count),dtype=float)

filterSeiten = np.full(fill_value=3/12,shape=(count),dtype=float)


filterMitte = randGen.standard_normal(count)**2

"""
diagonals = [filterEcken,filterSeiten[width:-width],filterEcken[width:-width],
     filterSeiten[1:-1],filterMitte,filterSeiten[1:-1],
    filterEcken[width:-width],filterSeiten[width:-width],filterEcken[width+1:-width-1],
     ]
"""
#https://de.mathworks.com/help/matlab/math/sparse-matrix-reordering.html

diagonals = np.array([filterEcken,filterSeiten,filterEcken,
     filterSeiten,filterMitte,filterSeiten,
    filterEcken,filterSeiten,filterEcken,
     ],copy=True)



offset = [-width-1,-width,-width+1,
     -1,0,1,
     width-1,width,width+1]


A=sparse.spdiags(
    diagonals,
    offset,count,count
    )
start = time()
A = sparse.csc_matrix(A)

print("Zeit für Conv: ",time()-start)
b = randGen.standard_normal(count)


start = time()
x=A.dot(b)
print("Time Mult A*b",time()-start)



#Invers Berechnen


start = time()


#approxInvB=sparse.spdiags(1/filterMitte,0,count,count)
#invB = sparse.linalg.inv(A)
#x = invB.dot(b)
#x,info = sparse.linalg.bicgstab(A,b,atol = 0.01)
x, istop, itn, normr  = sparse.linalg.lsmr(A,b,atol = 0.01)[:4]


#x, info = sparse.linalg.cg(A,b,atol = 0.01,maxiter=20)


print("Zeit für LG: ",time()-start )

#print("Info: ",istop, itn, normr)

m = np.mean(np.sqrt((A*x-b)**2+0.001**2))
print("Mean Error: ",m)


