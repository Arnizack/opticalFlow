from src.horn_schunck.setup_linear_system import *
import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy import sparse



def test_setup_linear_system1():
    img = np.zeros(shape=(3, 4, 5))
    deriv = np.full(shape=(3, 2, 5, 4), fill_value=1)
    A, b = setup_linear_system(img, img, deriv, 12)
    plt.matshow(A.todense())
    plt.show()

def test_setup_outer_array():
    expected_A = [1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0]
    A=setup_outer_array(4,5,1)
    print(expected_A==A)
    plt.matshow([A])
    plt.show()



def test_setup_inner_array():
    expected_A = [1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0]
    A = setup_inner_array(4, 5, 1)
    print(expected_A == A)
    plt.matshow([A])
    plt.show()

def test_setup_convolution():
    diags,offset = setup_diagonals_convolution(4,5,1)
    A = sparse.spdiags(diags,offset,5*4*2,5*4*2)
    A=A.todense()
    plt.matshow(A)
    plt.show()



def test_setup_linear_system2():
    width = 5
    height = 4
    img1 = 0
    img2 = 3
    I_x = 1
    I_y = 2
    I_t = img2-img1
    alpha = 12
    a = -alpha**2 /6
    b = -alpha**2 /12
    P=I_x**2+alpha**2
    Q=I_y**2+alpha**2
    R=I_x*I_y
    expected_A = [
        [Q, a, 0, 0, 0, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, a, Q, 0, 0, 0, b, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [a, b, 0, 0, 0, Q, a, 0, 0, 0, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [b, a, b, 0, 0, a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, b, a, b, 0, 0, a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, b, a, b, 0, 0, a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, b, a, 0, 0, 0, a, Q, 0, 0, 0, b, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, a, b, 0, 0, 0, Q, a, 0, 0, 0, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, b, a, b, 0, 0, a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, Q, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, b, a, 0, 0, 0, a, Q, 0, 0, 0, b, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, 0, 0, 0, Q, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, Q, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, Q, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, Q, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, 0, 0, 0, a, Q, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R],
        [R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, P, a, 0, 0, 0, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, P, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, P, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, P, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, P, 0, 0, 0, b, a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, 0, 0, 0, P, a, 0, 0, 0, a, b, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a, 0, 0, b, a, b, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a, 0, 0, b, a, b, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, 0, 0, 0, a, P, 0, 0, 0, b, a, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, 0, 0, 0, P, a, 0, 0, 0, a, b, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a, 0, 0, b, a, b, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a, 0, 0, b, a, b, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a, 0, 0, b, a, b],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, 0, 0, 0, a, P, 0, 0, 0, b, a],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, b, 0, 0, 0, P, a, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, b, 0, 0, a, P, a],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, R, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b, a, 0, 0, 0, a, P]
    ]

    expected_b = [-I_y*I_t]*20+[-I_x*I_t]*20

    np_img1 = np.full(fill_value=img1,shape=(1,height,width))
    np_img2 = np.full(fill_value=img2,shape=(1,height,width))
    np_I_x = np.full(fill_value=I_x,shape=(height,width))
    np_I_y = np.full(fill_value=I_y,shape=(height,width))
    derivative = np.array([[np_I_y,np_I_x]])
    actual_A,actual_b = setup_linear_system(np_img1,np_img2,derivative,alpha)

    actual_A = np.array(actual_A.todense())

    plt.matshow(actual_A)
    plt.matshow(expected_A)
    plt.matshow(actual_A==expected_A)
    plt.show()
    print(np.array_equal(actual_A,np.array(expected_A)))

def test_setup_linear_system3():
    derivative = np.zeros(shape=(1,2,5,4))
    img1 = np.zeros(shape=(1,5,4))
    derivative_y = np.array([
        [1,1,1,1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5, 5]
    ])

    derivative[0,0]=derivative_y
    img2 = np.zeros(shape=(1,5,4))

    np_img1 = np.full(fill_value=0, shape=(1, 5, 4))
    np_img2 = np.full(fill_value=3, shape=(1, 5, 4))
    np_I_x = np.full(fill_value=1, shape=(5, 4))
    derivative[0, 1] = np_I_x

    A,b = setup_linear_system(img1,img2,derivative,12)
    plt.matshow(A.todense())
    plt.show()

if __name__ == '__main__':
    #test_setup_linear_system1()
    test_setup_linear_system3()