#include"cpu_backend/flow/CrossBilateralMedianFilter.h"
#include"gtest/gtest.h"
#include"cpu_backend/flow/inner/BilateralMedian.h"
#include"cpu_backend/flow/inner/BilateralMedianSpeedup.h"

namespace cpu_backend
{
	TEST(BilateralMedianFilterTest, test1)
	{
		float img1[12] =
		{
			0.49302601, 0.69430977,
			0.33955504, 0.56177763,
			0.51197335, 0.83576093,
			0.31848839, 0.69716167,
			0.37070551, 0.54087005,
			0.25687337, 0.41850402
		};

		double log_occlusion[4] =
		{
			-21.91492715 , -2.63107706 ,
			-5.94522974 , -0.18044585
		};

		double flow[8] =
		{
			0.15555215, 0.18459567,
			0.18459567, 0.1249193,
			0.5367303 , 0.45425504,
			0.62003505, 0.38508236
		};

		double init_flow[8] =
		{
			0,0,
			0,0,
			0,0,
			0,0
		};

		double actual_result[8] = { 0 };

		double filter_influence = 6;
		double auxilary_influence = 2;
		double sigma_color = 4;
		double sigma_distance = 5;
		int filter_size = 3;

		BilateralMedian(flow, init_flow, log_occlusion, img1, filter_influence, auxilary_influence, sigma_distance, sigma_color,
			filter_size, 2, 2, 3, actual_result);
		
		double expected_result[8] =
		{
			0.18459567, 0.18459567, 0.18459567, 0.18459567, 0.62003505, 0.62003505,
			0.62003505, 0.62003505
		};

		for (int i = 0; i < 8; i++)
			EXPECT_NEAR(actual_result[i], expected_result[i], 0.00001);
	}
	TEST(BilateralMedianFilterTest, Speedup)
	{
		float img1[12] =
		{
			0.49302601, 0.69430977,
			0.33955504, 0.56177763,
			0.51197335, 0.83576093,
			0.31848839, 0.69716167,
			0.37070551, 0.54087005,
			0.25687337, 0.41850402
		};

		double log_occlusion[4] =
		{
			-21.91492715 , -2.63107706 ,
			-5.94522974 , -0.18044585
		};

		double flow[8] =
		{
			2, 3, 
			6, 5,
			8, 4, 
			7, 9
		};

		double init_flow[8] =
		{
			0,0,
			0,0,
			0,0,
			0,0
		};

		bool is_edge_map[8] =
		{
			false
		};

		double actual_result[8] = { 0 };

		double filter_influence = 6;
		double auxilary_influence = 2;
		double sigma_color = 4;
		double sigma_distance = 5;
		int bilateral_filter_size = 3;
		int median_filter_size = 3;

		//BilateralMedian(flow, init_flow, log_occlusion, img1, filter_influence, auxilary_influence, sigma_distance, sigma_color,
		//	bilateral_filter_size, 2, 2, 3, actual_result);

		BilateralMedianEdgeSpeedup(flow, init_flow, log_occlusion,
			img1, filter_influence, auxilary_influence, sigma_distance,
			sigma_color, is_edge_map, bilateral_filter_size, median_filter_size, 2, 2, 3, actual_result);

		double expected_result[8] =
		{
			3, 3,
			5, 5,
			8,7,
			7,8
		};

		for (int i = 0; i < 8; i++)
			EXPECT_NEAR(actual_result[i], expected_result[i], 0.00001);
	}
}