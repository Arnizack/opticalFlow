#include"gtest/gtest.h"

#include "cpu_backend/image/inner/convolution1D.h"

#include <iostream>
#include <algorithm>

namespace cpu_backend
{
	namespace testing
	{
		void delta_func(int* image, int width, int height, int* dst_X_gradient, int* dst_Y_gradient)
		{
			int kernel[3] = { 0, -1, 1 };

			int kernel_length = 3;
			int kernel_length_half = kernel_length / 2;

			#pragma omp parallel for
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					if (x == width - 1)
					{
						dst_X_gradient[width * y + x] = 0;
					}
					else
					{
						int sum = 0;

						for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
						{
							int x1 = _inner::CorrespondingX<Direction::X>(kernel_idx, kernel_length_half, x);
							int y1 = _inner::CorrespondingY<Direction::X>(kernel_idx, kernel_length_half, y);
							int img_val = _inner::GetValueAt<int, Padding::ZEROS>(x1, y1, width, height, image);
							sum += img_val * kernel[kernel_idx];
						}

						dst_X_gradient[width * y + x] = sum;
					}

					if (y == height - 1)
					{
						dst_Y_gradient[width * y + x] = 0;
					}
					else
					{
						int sum = 0;

						for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
						{
							int x1 = _inner::CorrespondingX<Direction::Y>(kernel_idx, kernel_length_half, x);
							int y1 = _inner::CorrespondingY<Direction::Y>(kernel_idx, kernel_length_half, y);
							int img_val = _inner::GetValueAt<int, Padding::ZEROS>(x1, y1, width, height, image);
							sum += img_val * kernel[kernel_idx];
						}

						dst_Y_gradient[width * y + x] = sum;
					}
				}
			}
		}

		TEST(ROFPreProcessing, Delta_Function_Test)
		{
			int arr[9];

			for (int i = 1; i < 10; i++)
			{
				arr[i - 1] = i;
			}

			int out_arr_x[9];
			int out_arr_y[9];

			int comp_x[9] ={1, 1, 0,
							1, 1, 0,
							1, 1, 0 };

			int comp_y[9] ={3, 3, 3,
							3, 3, 3,
							0, 0, 0 };

			delta_func(arr, 3, 3, out_arr_x, out_arr_y);

			for (int i = 0; i < 9; i++)
			{
				EXPECT_EQ(out_arr_x[i], comp_x[i]);
				EXPECT_EQ(out_arr_y[i], comp_y[i]);
			}
		}
    }
}