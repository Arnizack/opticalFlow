from src.utilities.penalty_functions.IPenalty import IPenalty
from src.sun_baker.setup_linear_system.convolution3x3 import convolution3x3
import scipy.sparse as sp
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def setup_linear_system(I_x: np.ndarray, I_y: np.ndarray, I_t: np.ndarray, guess_vu: np.ndarray, relax_vu: np.ndarray,
                        kernel: List[List[float]], width: int, height: int, lambda_r: float, lambda_k: float,
                        penalty_func: IPenalty) -> Tuple[sp.dia_matrix,np.ndarray]:
    """
    Solve:

    (I_x p''_D I_x + 2 lambda_R ) u + I_x p''_D I_y v - 2 lambda_K ker(u)
    = -I_x p'_D + I_x p''_D I_x du + I_x p''_D I_y dv + lambda_R 2 u_relax

    And

    (I_y p''_D I_y + 2 lambda_R ) v + I_x p''_D I_y u - 2 lambda_K ker(v)
    = -I_y p'_D + I_y p''_D I_y dv + I_x p''_D I_y du + lambda_R 2 v_relax

    with:
        I_x = derivative in x direction of the Image
        I_y = derivative in y direction of the Image
        u = x component of the flow
        v = y component of the flow
        du = guess for u
        dv = guess for v

        u_relax, v_relax = relaxation flow field
        lambda_R = weight for the relaxation flow field

        ker = smoothness kernel
        lambda_K = weight for the kernel

        p'_D first derivative of the penalty function at I_t + I_x du + I_y dv
        p''_D second derivative of the penalty function at I_t + I_x du + I_y dv

    A = |K+D_y R    |
        |R     K+D_x|

    b = |-I_y p'_D + I_y p''_D I_y dv + I_x p''_D I_y du + lambda_R 2 v_relax|
        |-I_x p'_D + I_x p''_D I_x du + I_x p''_D I_y dv + lambda_R 2 u_relax|

    K = - 2 lambda_K * mat(ker(x))
    D_y = diag(I_y p'_D I_y + 2 lambda_R)
    R = diag(I_x p''_D I_y)
    D_x = diag(I_x p'_D I_x + 2 lambda_R )


    :return: Diagonal Matrix , Vector
    """
    guess_v = guess_vu[:width * height]
    guess_u = guess_vu[width * height:]
    relax_v = relax_vu[:width * height]
    relax_u = relax_vu[width * height:]

    linearization_point = I_t + I_x * guess_u + I_y * guess_v
    first_derivative_penalty = penalty_func.get_first_derivative_at(linearization_point)
    second_derivative_penalty = penalty_func.get_second_derivative_at(linearization_point)

    kernel_matrix = - 2 * lambda_k * convolution3x3(kernel, width, height)
    diagonals_y = sp.spdiags(I_y * second_derivative_penalty * I_y + 2 * lambda_r, 0, width * height, width * height)
    diagonals_x = sp.spdiags(I_x * second_derivative_penalty * I_x + 2 * lambda_r, 0, width * height, width * height)
    rest_matrix = sp.spdiags(I_x * second_derivative_penalty * I_y, 0, width * height, width * height)



    A = sp.bmat([[kernel_matrix + diagonals_y, rest_matrix],
                 [rest_matrix, kernel_matrix + diagonals_x]], format="dia")

    b = np.empty(shape=(2 * width * height))
    b[:width * height] = -I_y * first_derivative_penalty + I_y * second_derivative_penalty * I_y * guess_v + \
                         I_x * second_derivative_penalty * I_y * guess_u + lambda_r * 2 * relax_v

    b[width * height:] = -I_x * first_derivative_penalty + I_x * second_derivative_penalty * I_x * guess_u + \
                         I_x * second_derivative_penalty * I_y * guess_v + lambda_r * 2 * relax_u

    return (A, b)
