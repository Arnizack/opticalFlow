from src.utilities.warp_grid import warp_image,warp_derivative
from src.utilities.image_access import open_image,show_image
from scipy import sparse
import scipy.sparse.linalg as splinalg
import numpy as np
import matplotlib.pyplot as plt

from time import time
from scipy.signal import medfilt2d
from src.horn_schunck.setup_linear_system import setup_linear_system
from src.utilities.cg_solver import cg
from pyamg.aggregation import smoothed_aggregation_solver

from src.horn_schunck.solver_settings import SolverSettings



def solve_layer(first_frame,second_frame, initial_flow_field, solver_settings):
    """

    :param first_frame: np.array(float) shape = (ColorChannel,Height,Width)
    :param second_frame: np.array(float) shape = (ColorChannel,Height,Width)
    :param first_frame_derivative: np.array(float) shape = (ColorChannel,Derivative_Direction,Height,Width)
    :param second_frame_derivative: np.array(float) shape = (ColorChannel,Derivative_Direction,Height,Width)
    :param initial_flow_field: np.array(float) (Flow_Direction, Height,Width)
    :param solver_settings: SolverSettings
    :return: np.array(float) (Flow_Direction, Height,Width)
    """
    #wrap second image
    second_frame_warped = warp_image(second_frame,initial_flow_field)

    A,b = setup_linear_system(first_frame,second_frame_warped,solver_settings)


    start = time()

    solver = solver_settings.solver
    if(solver=="lsmr"):

        x,info = splinalg.lsmr(A,b)[:2]
    elif(solver=="cg"):
        # A = sparse.csr_matrix(A)
        # multilevel = smoothed_aggregation_solver(A)

        # M = multilevel.aspreconditioner(cycle='AMLI')

        x, info = splinalg.cg(A, b, atol=0.001, maxiter=100)
        print("Diff Ax-b: ", np.mean((A.dot(x) - b) ** 2))
    elif(solver=="cg_own"):
        x, info = cg(A, b, maxiter=100)
    elif(solver=="bicgstab"):
        x,info =splinalg.bicgstab(A,b)
    elif(solver=="minres"):
        x,info = splinalg.minres(A,b,maxiter=100)[:2]
    elif(solver=="spsolve"):
        A = sparse.csr_matrix(A)
        x = splinalg.spsolve(A,b)

    print("Lg with", solver_settings.solver,": ",time()-start)
    print("Number of iterations: ", info, "\nMean error: ", (b - A.dot(x)).mean())

    width = first_frame.shape[2]
    height = first_frame.shape[1]
    x.shape = (2,height,width)

    flow = x+initial_flow_field
    if(solver_settings.median_filter_size > 0 ):
        flow[0] = medfilt2d(flow[0],solver_settings.median_filter_size)
        flow[1] = medfilt2d(flow[1],solver_settings.median_filter_size)


    return flow




