#pragma once
#include"filters/Gaussian1DFilter.h"
#include<memory>
#include"debug_helper/ImageLogger.h"
#include<type_traits>

namespace cpu_backend
{

	namespace _inner
	{
		template<class T>
		T ComputeStandardDeviation(int src_length, int dst_length)
		{
			T scale_factor = (T)dst_length / (T)src_length;
			return 1 / sqrt(2 * scale_factor);
		}


		template<class T>
		void DownScaleGaussianGrayScale(T* image, int width, int height, int dest_width, int dest_height, T* destination)
		{
			T scale_factor_width = (T)dest_width / (T)width;
			T scale_factor_height = (T)dest_height / (T)height;
			auto temp_x_blured = std::make_unique<std::vector<T>>(width * height);
			auto temp_xy_blured = std::make_unique<std::vector<T>>(width * height);

			T* ptr_temp_x_blured = temp_x_blured->data();
			T* ptr_temp_xy_blured = temp_xy_blured->data();

			T std_deviation_x = _inner::ComputeStandardDeviation<T>(width, dest_width);
			T std_deviation_y = _inner::ComputeStandardDeviation<T>(height, dest_height);

			Gaussian1DFilter<T, Direction::X, Padding::SYMMETRIC>(image, width, height, std_deviation_x, ptr_temp_x_blured);
			Gaussian1DFilter<T, Direction::Y, Padding::SYMMETRIC>(ptr_temp_x_blured, width, height, std_deviation_y, ptr_temp_xy_blured);
	
			T dest_to_src_scaler_x = (T)width / (T)dest_width;

			T dest_to_src_scaler_y = (T)height / (T)dest_height;

			T result = 0;

			for(int dest_y = 0; dest_y < dest_height; dest_y++)
			{
			
				int src_y = round( dest_to_src_scaler_y * (T)dest_y + 0.25* dest_to_src_scaler_y);
				for (int dest_x = 0; dest_x < dest_width; dest_x++)
				{
					int src_x = round(dest_to_src_scaler_x * (T)dest_x + 0.25 * dest_to_src_scaler_x);
					result = ptr_temp_xy_blured[src_y * width + src_x];
					destination[dest_y * dest_width + dest_x] = result;
				}
			}

		}

		template<class T>
		void DownScaleGaussianColorScale(T* image, int width, int height, int dest_width, int dest_height, int color_channel, T* destination)
		{
			const size_t in_wh = width * height;
			const size_t dest_wh = dest_width * dest_height;

			for (size_t z = 0; z < color_channel; z++)
			{
				DownScaleGaussianGrayScale<T>(image + (z * in_wh), width, height, dest_width, dest_height, destination + (z * dest_wh));
			}
		}
	}
}