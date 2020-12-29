#pragma once

#include <omp.h>
#include <cmath>

namespace cpu_backend
{
	namespace _rof_inner
	{
		void subtracted(float* arr1, float* arr2, const int& size)
		{
			//arr1 -= arr2

			for (int i = 0; i < size; i++)
			{
				arr1[i] -= arr2[i];
			}
		}

		float NormEuclidean(float* in, const int& size)
		{
			double norm = 0;

			#pragma omp parallel for reduction(+: norm)
			for (int i = 0; i < size; i++)
			{
				norm += in[i] * in[i];
			}

			return sqrt(norm);
		}

		float calculate_divergence_At(float* x_gradient, float* y_gradient, int* kernel, const int& kernel_length, const int& x, const int& y, const int& width, const int& height)
		{

			const int kernel_length_half = kernel_length / 2;

			float divergence = 0;

			if (x == width - 1)
			{
				divergence += -_inner::GetValueAt<float, Padding::ZEROS>(width - 2, y, width, height, x_gradient);
			}
			else
			{
				for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
				{
					int x1 = _inner::CorrespondingX<Direction::X>(kernel_idx, kernel_length_half, x);
					int y1 = _inner::CorrespondingY<Direction::X>(kernel_idx, kernel_length_half, y);
					float img_val = _inner::GetValueAt<float, Padding::ZEROS>(x1, y1, width, height, x_gradient);
					divergence += img_val * kernel[kernel_idx];
				}
			}

			if (y == height - 1)
			{
				divergence += -_inner::GetValueAt<float, Padding::ZEROS>(x, height - 2, width, height, y_gradient);
			}
			else
			{
				for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
				{
					int x1 = _inner::CorrespondingX<Direction::Y>(kernel_idx, kernel_length_half, x);
					int y1 = _inner::CorrespondingY<Direction::Y>(kernel_idx, kernel_length_half, y);
					float img_val = _inner::GetValueAt<float, Padding::ZEROS>(x1, y1, width, height, y_gradient);
					divergence += img_val * kernel[kernel_idx];
				}
			}

			return divergence;
		}

		float max_variation(float* current_x_gradient, float* current_y_gradient, float* next_x_gradient, float* next_y_gradient, const int& width, const int& height)
		{
			float max_variation = sqrt(pow(current_x_gradient[0], 2) + pow(current_y_gradient[0], 2));
			max_variation -= sqrt(pow(next_x_gradient[0], 2) + pow(next_y_gradient[0], 2));

			if (max_variation < 0)
				max_variation *= -1;

			for (int i = 0; i < width * height; i++)
			{
				float temp = sqrt(pow(current_x_gradient[i], 2) + pow(current_y_gradient[i], 2));

				temp -= sqrt(pow(next_x_gradient[i], 2) + pow(next_y_gradient[i], 2));

				if (temp < 0)
					temp *= -1;

				if (temp > max_variation)
				{
					max_variation = temp;
				}
			}

			return max_variation;
		}

		float total_variation(float* x_gradient, float* y_gradient, const int& width, const int& height)
		{
			float total_variation = 0;
			float temp = 0;

			#pragma omp parallel for reduction(+: total_variation)
			for (int y = 0; y < height; y++)
			{
				for (int x = 0; x < width; x++)
				{
					temp += powf(_inner::GetValueAt<float, Padding::ZEROS>(x, y, width, height, x_gradient), 2);
					temp += powf(_inner::GetValueAt<float, Padding::ZEROS>(x, y, width, height, y_gradient), 2);

					total_variation += sqrtf(temp);
				}
			}

			return total_variation;
		}

		float variance(float* matrix, const int& width, const int& height)
		{
			float size = width * height;

			float sum = 0;

			for (int i = 0; i < size; i++)
			{
				sum += matrix[i];
			}

			float mean = sum / size;

			sum = 0;

			for (int i = 0; i < size; i++)
			{
				sum += powf(matrix[i] - mean, 2);
			}

			return sum / size;
		}
	}
}