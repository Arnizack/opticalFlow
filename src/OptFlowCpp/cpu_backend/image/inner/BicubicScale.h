#pragma once
#include "BicubicInterpolate.h"

namespace cpu_backend
{
	namespace _inner
	{
		/*
		* 2D Bicubic Interpolater (upscaling)
		*/
		template<class T>
		inline void BicubicGrayScale(const T* img, T* destination,
			const size_t& input_width, const size_t& input_height, 
			const size_t& dst_width, const size_t& dst_height)
		{
			const double width_realtion = (double)dst_width / (double)input_width;
			const double height_relation = (double)dst_height / (double)input_height;

			float x_proj;
			float y_proj;

			//#pragma omp parallel for
			for (size_t y = 0; y < dst_height; y++) // column y
			{

				y_proj = (y - 0.5) / height_relation;

				for (size_t x = 0; x < dst_width; x++) // row x
				{
					x_proj = (x - 0.5) / width_realtion;

					destination[y * dst_width + x] = _inner::BicubicInerpolateAt<T>(x_proj, y_proj, img, input_width, input_height);
				}
			}
		}

		/*
		* 3D Bicubic Interpolater (upscaling)
		*/
		template<class T>
		inline void BicubicColorScale(const T* image, T* destination,
			const size_t& input_width, const size_t& input_height,
			const size_t& dst_width, const size_t& dst_height)
		{
			const size_t in_wh = input_width * input_height;
			const size_t dst_wh = dst_width * dst_height;

			for (size_t z = 0; z < 3; z++)
			{
				BicubicGrayScale<T>(image + (z * in_wh), destination + (z * dst_wh), input_width, input_height, dst_width, dst_height);
			}
		}
	}
}