#pragma once
#include"cpu_backend/sb_linearsystem/SunBakerLinearOp.h"
#include"cpu_backend/sb_linearsystem/SunBakerLSUpdater.h"
#include"cpu_backend/penalty/QuadraticPenalty.h"
#include"gtest/gtest.h"

namespace cpu_backend
{
	namespace testing
	{
		TEST(SunBakerTest, LinearOpTest1)
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

		TEST(SunBakerTest, LinearOpBuilderTest1)
		{
			auto ptr_deriv_calc = std::make_shared< DerivativeCalculator>();

			double lambda_kernel = 0.2;
			lambda_kernel *= lambda_kernel;
			double relaxation = 0.01;

			SunBakerLSUpdater updater(ptr_deriv_calc, lambda_kernel);

			float first_frame_data[20] =
			{
				0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
			};
			float second_frame_data[20] =
			{
				20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
			};

			std::array<const size_t, 2> shape = { 5,4 };
			auto first_frame = std::make_shared<Array<float, 2>>(shape, first_frame_data);
			auto second_frame = std::make_shared<Array<float, 2>>(shape, second_frame_data);

			auto ptr_penalty = std::make_shared<QuadraticPenalty>();
			updater.SetFramePair(first_frame, second_frame);
			updater.SetPenalty(ptr_penalty);
			double flow_data[40] = { 0 };
			/*
			{
				0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
				20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
			};*/

			

			std::array<const size_t, 3> flow_shape = { 2,5,4 };
			auto linearization_flow
				= std::make_shared<Array<double, 3>>(flow_shape, flow_data);

			
			updater.UpdateParameter(linearization_flow, relaxation);

			double expected_diag_y[20] =
			{
				10.908888888888903,
				10.908888888888878,
				10.908888888888885,
				10.908888888888903,
				37.57555555555554,
				37.57555555555554,
				37.57555555555558,
				37.57555555555554,
				32.01999999999999,
				32.02000000000003,
				32.01999999999997,
				32.01999999999999,
				37.57555555555557,
				37.57555555555555,
				37.57555555555554,
				37.57555555555557,
				10.908888888888885,
				10.90888888888887,
				10.90888888888891,
				10.908888888888885
			};

			double expected_diag_x[20] =
			{
				0.7005555555555565,
				2.367222222222225,
				2.367222222222221,
				0.7005555555555538,
				0.7005555555555526,
				2.36722222222222,
				2.3672222222222286,
				0.7005555555555569,
				0.7005555555555575,
				2.3672222222222294,
				2.36722222222222,
				0.7005555555555526,
				0.7005555555555588,
				2.3672222222222135,
				2.367222222222217,
				0.7005555555555606,
				0.7005555555555497,
				2.3672222222222192,
				2.3672222222222286,
				0.7005555555555544
			};

			double expected_rest[20]
			{
				2.722222222222226,
				5.055555555555555,
				5.055555555555554,
				2.7222222222222205,
				5.055555555555543,
				9.388888888888882,
				9.388888888888905,
				5.055555555555559,
				4.666666666666672,
				8.666666666666684,
				8.666666666666657,
				4.666666666666655,
				5.055555555555568,
				9.38888888888887,
				9.388888888888877,
				5.055555555555575,
				2.7222222222222103,
				5.055555555555547,
				5.055555555555568,
				2.7222222222222197
			};

			double expected_b_vec[40] =
			{
				-93.3333333333334,
				-93.33333333333329,
				-93.33333333333331,
				-93.3333333333334,
				-173.3333333333333,
				-173.3333333333333,
				-173.3333333333334,
				 -173.3333333333333,
				 -159.99999999999997,
				 -160.00000000000006,
				 -159.99999999999991,
				 -159.99999999999997,
				 -173.33333333333337,
				 -173.33333333333331,
				 -173.3333333333333,
				 -173.33333333333337,
				 -93.33333333333331,
				 -93.33333333333326,
				 -93.33333333333343,
				 -93.33333333333331,
				 -23.33333333333335,
				 -43.33333333333336,
				 -43.33333333333332,
				 -23.333333333333304,
				 -23.333333333333282,
				 -43.333333333333314,
				 -43.33333333333339,
				 -23.333333333333357,
				 -23.333333333333364,
				 -43.3333333333334,
				 -43.333333333333314,
				 -23.333333333333282,
				 -23.33333333333339,
				 -43.33333333333325,
				 -43.333333333333286,
				 -23.33333333333342,
				 -23.333333333333233,
				 -43.3333333333333,
				 -43.33333333333339,
				-23.333333333333314
			};

			double* actual_diag_x = updater._data_x->Data();
			double* actual_diag_y = updater._data_y->Data();
			double* actual_rest = updater._rest->Data();
			double* actual_b_vec = updater._desired_result->Data();

			double near = 0.001;

			for (int i = 0; i < 20; i++)
				EXPECT_NEAR(actual_diag_x[i], expected_diag_x[i], near);
			
			for (int i = 0; i < 20; i++)
				EXPECT_NEAR(actual_diag_y[i], expected_diag_y[i], near);

			for (int i = 0; i < 20; i++)
				EXPECT_NEAR(actual_rest[i], expected_rest[i], near);

			for (int i = 0; i < 40; i++)
				EXPECT_NEAR(actual_b_vec[i], expected_b_vec[i], near);


		}

		
	}
}