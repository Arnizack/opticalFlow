#pragma once
#include<array>

namespace cpu_backend
{
	/*
	x,y \in [0,1]
	*/
	inline void InterpolateBicubicScalars(float x, float y,
		float lower_left_corner, float lower_right_corner,
		float upper_left_corner, float upper_right_corner,
		float lower_left_corner_deriv_x, float lower_right_corner_deriv_x,
		float upper_left_corner_deriv_x, float upper_right_corner_deriv_x,
		float lower_left_corner_deriv_y, float lower_right_corner_deriv_y,
		float upper_left_corner_deriv_y, float upper_right_corner_deriv_y,
		float lower_left_corner_deriv_xy, float lower_right_corner_deriv_xy,
		float upper_left_corner_deriv_xy, float upper_right_corner_deriv_xy,
		float* destination

		)
	{
		/*
		Formular
		
		    | a_00 a_01 a_02 a_03 | 
		    | a_10 a_11 a_12 a_13 |
		A = | a_20 a_21 a_22 a_23 |
		    | a_30 a_31 a_32 a_33 |
		
		    | f_00   f_01   f_y_00  f_y_01  |
		    | f_10   f_11   f_y_10  f_y_11  |
		V = | f_x_00 f_x_01 f_xy_00 f_xy_01 |
		    | f_x_10 f_x_11 f_xy_10 f_xy_11 |
		
		    | 1  0  0  0 |
		K = | 0  0  1  0 |
		    |-3  3 -2 -1 |
		    | 2  -2 1  1 |

		A = K * V * K^T

		*/
	}
}