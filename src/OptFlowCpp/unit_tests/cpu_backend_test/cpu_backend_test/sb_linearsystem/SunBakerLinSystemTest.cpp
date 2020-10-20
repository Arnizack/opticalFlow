#pragma once
#include"cpu_backend/sb_linearsystem/SunBakerLinSystem.h"
#include"gtest/gtest.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(SunBakerLinearOpTest, ApplyToTest1)
		{
			double a_data[6] = { 0,1,2,3,4,5 };
			double b_data[6] = { 6,7,8,9,10,11 };
			double c_data[6] = { 12,13,14,15,16,17 };

			std::array<const size_t, 1> shape = { 6 };
			auto a_diags = std::make_shared<Array<double, 1>>(shape, a_data);
			auto b_diags = std::make_shared<Array<double, 1>>(shape, b_data);
			auto c_diags = std::make_shared<Array<double, 1>>(shape, c_data);

			double lambda_kernel = 2;

			SunBakerLinearOp linear_op(2, 3, a_diags, b_diags, c_diags, lambda_kernel);

			std::array<const size_t, 1> vec_shape = { 12 };

			double vec_data[12] = { 0, 1,2,3,4,5,6,7,8,9,10,11 };

			auto dst_array = std::make_shared<cpu_backend::Array<double, 1>>(vec_shape);
			auto vec_array = std::make_shared<cpu_backend::Array<double, 1>>(vec_shape, vec_data);

			linear_op.ApplyTo(dst_array, vec_array);

			/*
			69.0,  93.33333333, 117.33333333, 149.33333333,
			186.33333333, 226.66666667
			*/
			//lambda_kernel = 2
			double expected_data[12] = {
				69.0,  93.33333333, 117.33333333,
				149.33333333, 186.33333333, 226.66666667,
				47.0, 77.33333333, 101.33333333,
				139.33333333, 188.33333333, 234.66666667 };
			
			//lambda_kernel = 0
			/*
			double expected_data[12] = {
				72.0,  92.0, 116.0, 144.0, 176.0, 212.0,
				36.0,  62.0,  92.0, 126.0, 164.0, 206.0

			};*/

			double* actual_data = dst_array->Data();
			for (int i = 0; i < 12; i++)
			{
				EXPECT_NEAR(actual_data[i], expected_data[i], 0.0002);
			}
		}
	}
}