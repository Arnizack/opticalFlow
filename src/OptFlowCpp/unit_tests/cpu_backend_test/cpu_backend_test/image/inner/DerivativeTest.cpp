#include"gtest/gtest.h"
#include"cpu_backend/image/inner/DerivativeCalculator.h"
#include"utilities/image_helper/ImageHelper.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(DerivativeTest, test1)
		{
			float img[20]
			{
				0,   0,   0.1, 0,   0,
				0,   0.1, 0.2, 0.1, 0,
				0.1, 0.2, 0.4, 0.2, 0.1,
				0,   0.1, 0.2, 0.1, 0,

			};
			DerivativeCalculator<float> deriv_calc;
			imagehelper::Image deriv_x;
			deriv_x.width = 5;
			deriv_x.height = 4;
			deriv_x.color_count = 1;
			deriv_x.data = std::make_shared<std::vector<float>>(20);

			imagehelper::Image deriv_y;
			deriv_y.width = 5;
			deriv_y.height = 4;
			deriv_y.color_count = 1;
			deriv_y.data = std::make_shared<std::vector<float>>(20);


			deriv_calc.ComputeDerivativeX(img, deriv_x.width, deriv_x.height, deriv_x.data->data());
			deriv_calc.ComputeDerivativeY(img, deriv_y.width, deriv_y.height, deriv_y.data->data());

			float expected_deriv_x[20]
			{
				-8.33333333e-03,  6.66666667e-02,  0.00000000e+00, -6.66666667e-02,
				8.33333333e-03,  5.83333333e-02,  1.25000000e-01,  0.00000000e+00,
			   -1.25000000e-01, -5.83333333e-02,  5.00000000e-02,  1.91666667e-01,
				1.73472348e-18, -1.91666667e-01, -5.00000000e-02,  5.83333333e-02,
				1.25000000e-01,  0.00000000e+00, -1.25000000e-01, -5.83333333e-02
			};
			float expected_deriv_y[20]
			{
				-0.00833333,  0.05833333,  0.05      ,  0.05833333, -0.00833333,
				0.06666667,  0.125     ,  0.19166667,  0.125     ,  0.06666667,
				0.        , -0.00833333, -0.00833333, -0.00833333,  0.        ,
				-0.075     , -0.075     , -0.15      , -0.075     , -0.075
			};
			for (int i = 0; i < 20; i++)
			{
				float actual_x = deriv_x.data->data()[i];
				float actual_y = deriv_y.data->data()[i];
				EXPECT_NEAR(expected_deriv_x[i], actual_x, 0.001);
				EXPECT_NEAR(expected_deriv_y[i], actual_y, 0.001);


			}

		}
	}
}