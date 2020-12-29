#include"gtest/gtest.h"

#include "cpu_backend/image/inner/convolution1D.h"

#include <iostream>
#include <algorithm>

namespace cpu_backend
{
	namespace testing
	{
		template<class T, Padding padding, Direction direction>
		void Convolute1D_CutLastIteration(T* img, int width, int height,
			T* kernel, int kernel_length, T* destination, T cut_value)
		{
			int kernel_length_half = kernel_length / 2;

			#pragma omp parallel for
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					if (direction == Direction::Y && y == height - 1)
					{
						destination[width * y + x] = cut_value * _inner::GetValueAt<T, padding>(x, height - 2, width, height, img);
					}
					else if (direction == Direction::X && x == width - 1)
					{
						destination[width * y + x] = cut_value * _inner::GetValueAt<T, padding>(width - 2, y, width, height, img);
					}
					else
					{
						int sum = 0;
						for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
						{
							int x1 = _inner::CorrespondingX<direction>(kernel_idx, kernel_length_half, x);
							int y1 = _inner::CorrespondingY<direction>(kernel_idx, kernel_length_half, y);
							T img_val = _inner::GetValueAt<T, padding>(x1, y1, width, height, img);
							sum += img_val * kernel[kernel_idx];
						}
						destination[width * y + x] = sum;
					}
				}
			}
		}

		float calculate_divergence_At(int* x_gradient, int* y_gradient, int* kernel, int kernel_length, int x, int y, int& width, int& height)
		{

			int kernel_length_half = kernel_length / 2;

			float divergence = 0;

			if (x == width - 1)
			{
				divergence += -_inner::GetValueAt<int, cpu_backend::Padding::ZEROS>(width - 2, y, width, height, x_gradient);
			}
			else
			{
				for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
				{
					int x1 = _inner::CorrespondingX<Direction::X>(kernel_idx, kernel_length_half, x);
					int y1 = _inner::CorrespondingY<Direction::X>(kernel_idx, kernel_length_half, y);
					float img_val = _inner::GetValueAt<int, Padding::ZEROS>(x1, y1, width, height, x_gradient);
					divergence += img_val * kernel[kernel_idx];
				}
			}

			if (y == height - 1)
			{
				divergence += -_inner::GetValueAt<int, cpu_backend::Padding::ZEROS>(x, height - 2, width, height, y_gradient);
			}
			else
			{
				for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
				{
					int x1 = _inner::CorrespondingX<Direction::Y>(kernel_idx, kernel_length_half, x);
					int y1 = _inner::CorrespondingY<Direction::Y>(kernel_idx, kernel_length_half, y);
					float img_val = _inner::GetValueAt<int, Padding::ZEROS>(x1, y1, width, height, y_gradient);
					divergence += img_val * kernel[kernel_idx];
				}
			}

			return divergence;
		}


		void divergence_func(int* x_gradient, int* y_gradient, int width, int height, int* dst)
		{
			int kernel[2] = { -1,1 };

			int kernel_length = 2;
			int kernel_length_half = 1;

			#pragma omp parallel for
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					//int sum = 0;

					//if (x == width - 1)
					//{
					//	sum += -_inner::GetValueAt<int, cpu_backend::Padding::ZEROS>(width - 2, y, width, height, x_gradient);
					//}
					//else
					//{
					//	for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
					//	{
					//		int x1 = _inner::CorrespondingX<Direction::X>(kernel_idx, kernel_length_half, x);
					//		int y1 = _inner::CorrespondingY<Direction::X>(kernel_idx, kernel_length_half, y);
					//		int img_val = _inner::GetValueAt<int, Padding::ZEROS>(x1, y1, width, height, x_gradient);
					//		sum += img_val * kernel[kernel_idx];
					//	}
					//}

					//if (y == height - 1)
					//{
					//	sum += -_inner::GetValueAt<int, cpu_backend::Padding::ZEROS>(x, height - 2, width, height, y_gradient);
					//}
					//else
					//{
					//	for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
					//	{
					//		int x1 = _inner::CorrespondingX<Direction::Y>(kernel_idx, kernel_length_half, x);
					//		int y1 = _inner::CorrespondingY<Direction::Y>(kernel_idx, kernel_length_half, y);
					//		int img_val = _inner::GetValueAt<int, Padding::ZEROS>(x1, y1, width, height, y_gradient);
					//		sum += img_val * kernel[kernel_idx];
					//	}
					//}

					//dst[width * y + x] = sum;

					dst[width * y + x] = calculate_divergence_At(x_gradient, y_gradient, kernel, kernel_length, x, y, width, height);
				}
			}
		}
		
		TEST(ROFPreProcessing, Convolution_Test)
		{
			int arr[9];

			for (int i = 1; i < 10; i++)
			{
				arr[i - 1] = i;
			}

			int kernel[] = { -1, 1 };

			int out_arr_x[9];
			int out_arr_y[9];

			Convolute1D_CutLastIteration<int, Padding::ZEROS, Direction::X>(arr, 3, 3, kernel, 2, out_arr_x, -1);
			Convolute1D_CutLastIteration<int, Padding::ZEROS, Direction::Y>(arr, 3, 3, kernel, 2, out_arr_y, -1);

			int comp_x[9] = { 1, 1, -2,
								4, 1, -5,
								7, 1, -8 };

			int comp_y[9] = { 1, 2, 3,
								3, 3, 3,
								-4, -5, -6 };

			for (int i = 0; i < 9; i++)
			{
				EXPECT_EQ(out_arr_x[i], comp_x[i]);
				EXPECT_EQ(out_arr_y[i], comp_y[i]);
			}
		}

		TEST(ROFPreProcessing, Divergence_Function_Test)
		{
			int arr[9];

			for (int i = 1; i < 10; i++)
			{
				arr[i-1] = i;
			}

			int kernel[] = { -1, 1 };

			int out_arr_x[9];
			int out_arr_y[9];

			Convolute1D_CutLastIteration<int, Padding::ZEROS, Direction::X>(arr, 3, 3, kernel, 2, out_arr_x, -1);
			Convolute1D_CutLastIteration<int, Padding::ZEROS, Direction::Y>(arr, 3, 3, kernel, 2, out_arr_y, -1);

			int comp_x[9] =	  { 1, 1, -2,
								4, 1, -5,
								7, 1, -8 };

			int comp_y[9] =   { 1, 2, 3,
								3, 3, 3,
								-4, -5, -6 };

			std::cout << "In:\n";

			for (int i = 0; i < 9; i++)
			{
				if (i % 3 == 0)
				{
					std::cout << '\n';
				}

				std::cout << arr[i] << ' ';
			}

			std::cout << "\n\nOut:\n";

			for (int i = 0; i < 9; i++)
			{
				if (i % 3 == 0)
				{
					std::cout << '\n';
				}

				std::cout << out_arr_x[i] << ' ';
			}

			for (int i = 0; i < 9; i++)
			{
				EXPECT_EQ(out_arr_x[i], comp_x[i]);
				EXPECT_EQ(out_arr_y[i], comp_y[i]);
			}

			int out_arr_div[9];

			divergence_func(arr, arr, 3, 3, out_arr_div);

			for (int i = 0; i < 9; i++)
			{
				EXPECT_EQ(out_arr_x[i] + out_arr_y[i], out_arr_div[i]);
			}
		}
	}
}