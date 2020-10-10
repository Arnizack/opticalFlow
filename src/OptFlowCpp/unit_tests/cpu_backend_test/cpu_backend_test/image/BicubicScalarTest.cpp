#include"gmock/gmock.h"
#include"cpu_backend/image/warper/InterpolateBicubic.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(BicubicScalarTest, test1)
		{
			float lower_left_corner = 1; 
			float lower_right_corner = 2;
			float upper_left_corner = 3; 
			float upper_right_corner = 4;
			float lower_left_corner_deriv_x = 5; 
			float lower_right_corner_deriv_x = 6;
			float upper_left_corner_deriv_x = 7; 
			float upper_right_corner_deriv_x = 8;
			float lower_left_corner_deriv_y = 9; 
			float lower_right_corner_deriv_y = 10;
			float upper_left_corner_deriv_y = 11; 
			float upper_right_corner_deriv_y = 12;
			float lower_left_corner_deriv_xy = 13; 
			float lower_right_corner_deriv_xy = 14;
			float upper_left_corner_deriv_xy = 15; 
			float upper_right_corner_deriv_xy  = 16;
			std::array<float, 16> dst;
			int count = 1;
			for(int i = count; i>0 ; i--)
			InterpolateBicubicScalars(dst[0], lower_right_corner,
				upper_left_corner, upper_right_corner, 
				lower_left_corner_deriv_x, lower_right_corner_deriv_x, 
				upper_left_corner_deriv_x, upper_right_corner_deriv_x,
				lower_left_corner_deriv_y, lower_right_corner_deriv_y, 
				upper_left_corner_deriv_y, upper_right_corner_deriv_y, 
				lower_left_corner_deriv_xy, lower_right_corner_deriv_xy, 
				upper_left_corner_deriv_xy, upper_right_corner_deriv_xy, dst.data());

			InterpolateBicubicScalars(lower_left_corner, lower_right_corner,
				upper_left_corner, upper_right_corner,
				lower_left_corner_deriv_x, lower_right_corner_deriv_x,
				upper_left_corner_deriv_x, upper_right_corner_deriv_x,
				lower_left_corner_deriv_y, lower_right_corner_deriv_y,
				upper_left_corner_deriv_y, upper_right_corner_deriv_y,
				lower_left_corner_deriv_xy, lower_right_corner_deriv_xy,
				upper_left_corner_deriv_xy, upper_right_corner_deriv_xy, dst.data());

			/*
			      | a_00 a_01 a_02 a_03 | 
			      | a_10 a_11 a_12 a_13 |
			DST = | a_20 a_21 a_22 a_23 |
			      | a_30 a_31 a_32 a_33 |
			
			    | f_00   f_01   f_y_00  f_y_01  |
			    | f_10   f_11   f_y_10  f_y_11  |
			V = | f_x_00 f_x_01 f_xy_00 f_xy_01 |
			    | f_x_10 f_x_11 f_xy_10 f_xy_11 |

				| 1 2 9  10 |
			  =	| 3 4 11 12 |
				| 5 6 13 14 |
				| 7 8 15 16 |

			
			    | 1  0  0  0 |
			K = | 0  0  1  0 |
			    |-3  3 -2 -1 |
			    | 2  -2 1  1 |
			
			DST = K * V * K^T

			DST = [[  1,   9, -25,  17],
				   [  5,  13, -37,  25],
				   [-11, -35,  99, -67],
				   [  8,  24, -68,  46]]

			*/

			float expected[16] =
			{
				  1,   9, -25,  17,
				  5,  13, -37,  25,
				- 11, -35,  99, -67,
				  8,  24, -68,  46
			};
			
			for (int i = 0; i < 16; i++)
			{
				EXPECT_EQ(expected[i], dst[i]);
			}
		}

		TEST(BicubicScalarTest, test2)
		{
			float scalars[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
			float x = 0.5;
			float y = 0.8;
			/*
				  | a_00 a_01 a_02 a_03 |
				  | a_10 a_11 a_12 a_13 |
			S =   | a_20 a_21 a_22 a_23 |
				  | a_30 a_31 a_32 a_33 |

				| 1 2 9  10 |
			  =	| 3 4 11 12 |
				| 5 6 13 14 |
				| 7 8 15 16 |


										|1  |
			result = 	[1,x,x^2,x^3] S |y  |
										|y^2|
										|y^3|

			*/

			float expected = 23.016000000000002;

			float actual = BicubicApplyScalars(scalars, x, y);

			EXPECT_NEAR(expected, actual, 0.00001);
			
		}
	}
}