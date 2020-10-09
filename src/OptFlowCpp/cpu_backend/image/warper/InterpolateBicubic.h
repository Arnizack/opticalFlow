#pragma once
#include<array>
#include"cblas.h"


namespace cpu_backend
{
	/*
	x,y \in [0,1]
	*/
	inline void InterpolateBicubicScalars(float lower_left_corner, float lower_right_corner,
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
	   DST = | a_20 a_21 a_22 a_23 |
			 | a_30 a_31 a_32 a_33 |

		   | f_00   f_01   f_y_00  f_y_01  |
		   | f_10   f_11   f_y_10  f_y_11  |
	   V = | f_x_00 f_x_01 f_xy_00 f_xy_01 |
		   | f_x_10 f_x_11 f_xy_10 f_xy_11 |

		   | 1  0  0  0 |
	   K = | 0  0  1  0 |
		   |-3  3 -2 -1 |
		   | 2  -2 1  1 |

	   DST = K * V * K^T
	   */
		 float& a = lower_left_corner;
		 float& b = lower_right_corner;
		 float& c = lower_left_corner_deriv_y;
		 float& d = lower_right_corner_deriv_y;

		 float& e = upper_left_corner; 
		 float& f = upper_right_corner;
		 float& g = upper_left_corner_deriv_y;
		 float& h = upper_right_corner_deriv_y;

		 float& i = lower_left_corner_deriv_x;
		 float& j = lower_right_corner_deriv_x;
		 float& k = lower_left_corner_deriv_xy;
		 float& l = lower_right_corner_deriv_xy;

		 float& m = upper_left_corner_deriv_x;
		 float& n = upper_right_corner_deriv_x;
		 float& o = upper_left_corner_deriv_xy;
		 float& p = upper_right_corner_deriv_xy;

		 destination[0] = a;
		 destination[1] = c;
		 destination[2] = -3.0*a + 3.0*b - 2.0*c - d;
		 destination[3] = 2.0* a - 2.0* b + c + d;

		 destination[4] = i;
		 destination[5] = k;
		 destination[6] = -3.0 * i + 3.0 * j - 2.0* k - l;
		 destination[7] = 2.0* i - 2.0* j + k + l;

		 destination[8] = -3.0* a + 3.0 *e - 2.0*i - m;
		 destination[9] = -3.0 * c + 3.0 * g - 2.0* k - o;
		 destination[10] = 3.0 * d - 3.0 * h + 2.0* l - 3.0 * (-3.0 * a + 3.0 * e - 2.0* i - m) + 3.0 * (-3.0 * b + 3.0 * f - 2.0* j - n) - 2.0* (-3.0 * c + 3.0 * g - 2.0* k - o) + p;
		 destination[11] = -3.0 * c -3.0* d +3.0* g +3.0* h - 2.0* k - 2.0* l + 2.0* (-3.0* a +3.0* e - 2.0* i - m) - 2.0* (-3.0* b + 3.0* f - 2.0* j - n) - o - p;

		 destination[12] = 2.0* a - 2.0* e + i + m;
		 destination[13] = 2.0* c - 2.0* g + k + o;
		 destination[14] = -2.0* d + 2.0* h - l - 3.0* (2.0* a - 2.0* e + i + m) + 3.0* (2.0* b - 2.0* f + j + n) - 2.0* (2.0* c - 2.0* g + k + o) - p;
		 destination[15] = 2.0*c + 2.0*d - 2.0*g - 2.0*h + k + l + 2.0*(2.0*a - 2.0*e + i + m) - 2.0*(2.0*b - 2.0*f + j + n) + o + p;
		

		//{ {a, c, -3 a + 3 b - 2 c - d, 2 a - 2 b + c + d}, { i, k, -3 i + 3 j - 2 k - l, 2 i - 2 j + k + l }, { -3 a + 3 e - 2 i - m, -3 c + 3 g - 2 k - o, 3 d - 3 h + 2 l - 3 (-3 a + 3 e - 2 i - m) + 3 (-3 b + 3 f - 2 j - n) - 2 (-3 c + 3 g - 2 k - o) + p, -3 c - 3 d + 3 g + 3 h - 2 k - 2 l + 2 (-3 a + 3 e - 2 i - m) - 2 (-3 b + 3 f - 2 j - n) - o - p }, 
		
	}
}