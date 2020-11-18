import scipy.sparse.linalg as splin
from scipy import sparse
import numpy as np


def cg(A, b, M=None, x=None, tol=1e-05, maxiter=100):
    """
    Calculates x wich solves approximatly b = Ax

    :param A: input Matrix b = Ax
    :param b: input Vector
    :type b: Same row count as A
    :param M: input Matrix to precondition A, should be A^-1 or None
    :param x: input Vector to start the search with, standard 0-vector with A rows count
    :param tol: defines who precise x shall be
    :param maxiter: maximal number of iterations
    :return: x and number of needed iterations
    """

    x, M = setup_x(A, x, M)

    #start values
    r = b - A.dot(x) #r = r_k current, r_temp = r_k+1 next
    h = M.dot(r) #h = h_k current, h_temp = h_k+1 next
    d = r

    for i in range(maxiter):
        z = A.dot(d)
        #minimize x in direction d
        a = r.T.dot(h) / (d.T.dot(z))
        x = x + (a * d)
        r_temp = r - (a * z)
        if np.linalg.norm(r_temp) < tol:
            return (x, i)
        h_temp = M.dot(r_temp)
        beta = feltcher_reeves(r, r_temp, h, h_temp)
        d = h_temp + (beta * d)
        r = r_temp
        h = h_temp

    return (x, maxiter)


def feltcher_reeves(r, r_k, h, h_k):
    """
    calculates the feltcher reeves kernel for updated d
    :param r: current r
    :param r_k: next r
    :param h: current h
    :param h_k: next h
    :return: b in R
    """
    b = r_k.T.dot(h_k)
    b /= r.T.dot(h)
    return b


def setup_x(A, x, M):
    """
    setups x and M if not specified
    :param A: Matix
    :param x: Vector
    :param M: Matrix
    :return: x and M
    """
    row, column = A.shape
    if x == None:
        x = np.zeros(column)
    if M == None:
        M = sparse.identity(row)
    return (x, M)
