# Explanation Linear Problem

## Goal
Solve:

(I_x p''_D I_x + 2 lambda_R ) u + I_x p''_D I_y v - 2 lambda_K ker(u)
= -I_x p'_D + I_x p''_D I_x du + I_x p''_D I_y dv + lambda_R 2 u_relax

And

(I_y p''_D I_y + 2 lambda_R ) v + I_x p''_D I_y u - 2 lambda_K ker(v)
= -I_y p'_D + I_y p''_D I_y dv + I_x p''_D I_y du + lambda_R 2 v_relax

with:
    I_x = derivative in x direction of the first image
    I_y = derivative in y direction of the first image
    I_t = second image - first image
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

## Solution

A = |K+D_y R    |
    |R     K+D_x|

b = |-I_y p'_D + I_y p''_D I_y dv + I_x p''_D I_y du + lambda_R 2 v_relax|
    |-I_x p'_D + I_x p''_D I_x du + I_x p''_D I_y dv + lambda_R 2 u_relax|

K = - 2 lambda_K * mat(ker(x))
D_y = diag(I_y p''_D I_y + 2 lambda_R)
R = diag(I_x p''_D I_y)
D_x = diag(I_x p''_D I_x + 2 lambda_R )

## Implementation

### SunBakerLSBuilder

#### SetFramePair(PtrGrayImg first_image, PtrGrayImg second_image)

I_y = deriv_y(first_image)
I_x = deriv_x(second_image)
I_t = second_image - first_image

#### UpdateParameter(PtrFlowField linearization_points, double relaxation)

compute p'_D, p''_D 

#### Update()

compute D_y, D_x, R, b

:return: Diagonal Matrix , Vector


## SunBaker