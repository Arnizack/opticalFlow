#pragma once
#include "ROFPreProcessing.h"

#include "ROFHelper.h"
#include "../../Base.h"

#include <omp.h>
#include <vector>
#include <cmath>
#include <iostream>

namespace cpu_backend
{
	using PtrImage = std::shared_ptr<core::IArray<float, 3>>;


	ROFPreProcessing::ROFPreProcessing(std::shared_ptr<ROFPreProcessingSettings> settings,
		std::shared_ptr<core::IArrayFactory<float, 2>> arr_factory2D, std::shared_ptr<core::IArrayFactory<float, 3>> arr_factory3D, 
		std::shared_ptr<core::IStatistics<float>> statistic,
		std::shared_ptr<core::IArithmeticVector<float, 2>> arith_vector)
		: _iter(settings->iter), _lambda(settings->lambda), _tau(settings->tau), _sub_factor(settings->sub_factor),
		_arr_factory2D(arr_factory2D), _arr_factory3D(arr_factory3D), _statistic(statistic),
		_arith_vector(arith_vector)
	{}

	std::shared_ptr<core::IArray<float, 3>> ROFPreProcessing::Process(PtrImage img)
	{
		/*
		* Refernce: https://www.uni-muenster.de/AMM/num/Vorlesungen/MathemBV_SS16/literature/Chambolle2004.pdf
		* 
		* See section 9 of the refenced paper, for an overview
		* 
		* Image denoising Algorithm
		*/

		const size_t color_count = img->Shape[0];
		const size_t height = img->Shape[1];
		const size_t width = img->Shape[2];
		const size_t width_height = width * height;

		auto img_ptr = std::static_pointer_cast<Array<float, 3>>(img)->Data();

		auto output_img = std::static_pointer_cast<Array<float, 3>>(_arr_factory3D->Zeros({ color_count, height, width }));
		auto output_img_ptr = output_img->Data();

		//Every color channel is calculated seperatly
		for (int i = 0; i < color_count; i++)
		{
			std::cout << "Color channel " << i << '\n';

			//setup
			int offset = width_height * i;

			img_ptr += offset;

			float img_variance = _rof_inner::variance(img_ptr, width, height);
			img_variance = 0.1f;

			std::vector<float> current_matrix(width_height, 0);
			float* current_matrix_data = current_matrix.data();


			//calculates the subtraction matrix
			pi_gradient_descend(img_ptr, img_variance, width, height, current_matrix_data);


			//normalize current_matrix
			float temp_norm = _rof_inner::NormEuclidean(current_matrix_data, width_height);

			#pragma omp parallel for
			for (int j = 0; j < width_height; j++)
			{
				current_matrix_data[j] /= temp_norm;
			}

			std::cout << "denoised calculated\n";


			//final output values
			#pragma omp parallel for
			for (int j = 0; j < width_height; j++)
			{
				(*output_img)[offset + j] = img_ptr[j] - (_sub_factor * current_matrix_data[j]);
			}

			std::cout << "Writting worked\n";
		}

		return output_img;
	}

	void ROFPreProcessing::pi_gradient_descend(float* img, const float& img_var, const int& width, const int& height, float* destination)
	{
		/*
		* calculates the subtarction matrix
		* 
		* See section 3 of the refenced paper
		* 
		* DEPENDS ON _LAMBDA
		*/

		//setup
		std::vector<float> current_x_gradient(width * height, 0);
		std::vector<float> current_y_gradient(width * height, 0);

		std::vector<float> next_x_gradient(width * height, 0);
		std::vector<float> next_y_gradient(width * height, 0);

		float* current_x_gradient_data = current_x_gradient.data();
		float* current_y_gradient_data = current_y_gradient.data();

		float* next_x_gradient_data = next_x_gradient.data();
		float* next_y_gradient_data = next_y_gradient.data();

		float current_total_variation = _rof_inner::total_variation(current_x_gradient_data, current_y_gradient_data, width, height);

		for (int i = 0; i < _iter; i++)
		{
			//update gradients
			calculate_next_gradients(current_x_gradient_data, current_y_gradient_data, img, width, height, next_x_gradient_data, next_y_gradient_data);

			float next_total_variation = _rof_inner::total_variation(next_x_gradient_data, next_y_gradient_data, width, height);

			float max_var = _rof_inner::max_variation(current_x_gradient_data, current_y_gradient_data, next_x_gradient_data, next_y_gradient_data, width, height);

			current_x_gradient_data = next_x_gradient_data;
			current_y_gradient_data = next_y_gradient_data;

			//change to maximum variation beetween current and next
			if (current_total_variation - next_total_variation < 0.01 && current_total_variation - next_total_variation > -0.01)
			{
				//break;
			}

			current_total_variation = next_total_variation;

			//update output to update lambda 
			divergence_func(current_x_gradient_data, current_y_gradient_data, width, height, destination);

			//update lambda
			_lambda *= (img_var * width * height) / _rof_inner::NormEuclidean(destination, width * height);

			//iteration stopper
			if (max_var < 0.1)
			{
				break;
			}
		}

		//final output
		#pragma omp parallel for
		for (int i = 0; i < width * height; i++)
		{
			destination[i] *= _lambda;
		}
	}

	void ROFPreProcessing::calculate_next_gradients(float* current_x_gradient, float* current_y_gradient, float* img, const int& width, const int& height, float* destination_x_gradient, float* destination_y_gradient)
	{
		/*
		* calculates the new gradient matrices
		* 
		* current != destination
		* because values from current are needed which would be overwritten
		*/

		int kernel[2] = { -1,1 };
		int kernel_length = 2;


		#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				//temp values for the current gradients | delta(div p - img / lambda)
				float temp_x_gradient_at = 0;
				float temp_y_gradient_at = 0;

				//factored img | img / lambda
				const float factored_img_at = _inner::GetValueAt<float, Padding::ZEROS>(x, y, width, height, img) / _lambda;

				//divergence value ( div p ) at x, y
				float divergence = _rof_inner::calculate_divergence_At(current_x_gradient, current_y_gradient, kernel, kernel_length, x, y, width, height);
				divergence -= factored_img_at;

				//calculates the total variation |delta(div p - img / lambda)| at x, y
				float temp_total_variation = 0;

				//condition from delta function
				if (x == width - 1)
				{
					temp_x_gradient_at = 0;
				}
				else
				{
					//divergence value ( div p ) at x+1, y
					float divergence_right = _rof_inner::calculate_divergence_At(current_x_gradient, current_y_gradient, kernel, kernel_length, x + 1, y, width, height);
					divergence_right -= _inner::GetValueAt<float, Padding::ZEROS>(x + 1, y, width, height, img) / _lambda;

					//gradient delta(div p - img / lambda) for X direction
					temp_x_gradient_at = divergence_right - divergence;

					//total variation counter
					temp_total_variation += powf(temp_x_gradient_at, 2);
				}

				//condition from delta function
				if (y == height - 1)
				{
					temp_y_gradient_at = 0;
				}
				else
				{
					//divergence value ( div p ) at x, y+1
					float divergence_below = _rof_inner::calculate_divergence_At(current_x_gradient, current_y_gradient, kernel, kernel_length, x, y + 1, width, height);
					divergence_below -= _inner::GetValueAt<float, Padding::ZEROS>(x, y + 1, width, height, img) / _lambda;

					//gradient delta(div p - img / lambda) for Y direction
					temp_y_gradient_at = divergence_below - divergence;

					//total variation counter
					temp_total_variation += powf(temp_y_gradient_at, 2);
				}

				//totalvariation update
				float total_variation = sqrtf(temp_total_variation);

				//calculate garient (p + tau * delta(div p - img / lambda)) / (1+ tau(|delta(div p - img / lambda)|))
				destination_x_gradient[width * y + x] = _inner::GetValueAt<float, Padding::ZEROS>(x, y, width, height, current_x_gradient) + (_tau * temp_x_gradient_at);
				destination_x_gradient[width * y + x] /= 1 + (_tau * total_variation);

				destination_y_gradient[width * y + x] = _inner::GetValueAt<float, Padding::ZEROS>(x, y, width, height, current_y_gradient) + (_tau * temp_y_gradient_at);
				destination_y_gradient[width * y + x] /= 1 + (_tau * total_variation);
			}
		}
	}

	void ROFPreProcessing::divergence_func(float* x_gradient, float* y_gradient, const int& width, const int& height, float* dst)
	{
		/*
		* Calculates the divergence beetween to gradients
		*/
		int kernel[2] = { -1,1 };

		int kernel_length = 2;

		#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float divergence = _rof_inner::calculate_divergence_At(x_gradient, y_gradient, kernel, kernel_length, x, y, width, height);

				dst[width * y + x] = divergence;
			}
		}
	}

	void ROFPreProcessing::delta_func(float* image, const int& width, const int& height, float* dst_X_gradient, float* dst_Y_gradient)
	{
		/*
		* Splits a single matix into x and y gradients
		*/

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
					float sum = 0;

					for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
					{
						int x1 = _inner::CorrespondingX<Direction::X>(kernel_idx, kernel_length_half, x);
						int y1 = _inner::CorrespondingY<Direction::X>(kernel_idx, kernel_length_half, y);
						float img_val = _inner::GetValueAt<float, Padding::ZEROS>(x1, y1, width, height, image);
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
					float sum = 0;

					for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
					{
						int x1 = _inner::CorrespondingX<Direction::Y>(kernel_idx, kernel_length_half, x);
						int y1 = _inner::CorrespondingY<Direction::Y>(kernel_idx, kernel_length_half, y);
						float img_val = _inner::GetValueAt<float, Padding::ZEROS>(x1, y1, width, height, image);
						sum += img_val * kernel[kernel_idx];
					}

					dst_Y_gradient[width * y + x] = sum;
				}
			}
		}
	}
}