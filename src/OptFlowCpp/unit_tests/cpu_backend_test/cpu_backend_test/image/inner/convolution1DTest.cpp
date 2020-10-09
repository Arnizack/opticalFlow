#pragma once
#include"gtest/gtest.h"
#include"cpu_backend/image/inner/convolution1D.h"
#include<array>

namespace cpu_backend
{
	namespace testing
	{
		TEST(InnerConv1DTest, XDirection)
		{
			std::array<float, 16> img = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
			std::array<float, 3> kernel = { 1,2,3 };
			std::array<float, 16> actual;
			Convolute1D<float, Padding::ZEROS, Direction::X>(img.data(), 4, 4, kernel.data(), 3, actual.data());
			std::array<float, 16> expected = 
			{ 3, 8, 14, 8,
			23, 32, 38, 20,
			43, 56, 62, 32,
			63, 80, 86, 44 };

			for (int i = 0; i < 16; i++)
				EXPECT_EQ(actual[i], expected[i]);

		}

		TEST(InnerConv1DTest, YDirection)
		{
			std::array<float, 16> img = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
			std::array<float, 3> kernel = { 1,2,3 };
			std::array<float, 16> actual;
			Convolute1D<float, Padding::ZEROS, Direction::Y>(img.data(), 4, 4, kernel.data(), 3, actual.data());
			std::array<float, 16> expected =
			{ 12, 17, 22, 27,
			 32, 38, 44, 50,
			 56, 62, 68, 74,
			 32, 35, 38, 41 };

			for (int i = 0; i < 16; i++)
				EXPECT_EQ(actual[i], expected[i]);

		}
	}
}