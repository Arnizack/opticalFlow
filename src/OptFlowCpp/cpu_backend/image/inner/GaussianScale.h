#pragma once
#include"DownScaleGaussianGrayScale.h"
namespace cpu_backend
{
	namespace _inner
	{
		template<typename T>
		inline void Gaussian2DScale(const T* img, T* destination,
			const size_t& input_width, const size_t& input_height,
			const size_t& dst_width, const size_t& dst_height)
		{
			DownScaleGaussianGrayScale<T>(img, input_width, input_height,
				dst_width, dst_height, destination);
		}

		template<typename T>
		inline void GaussianFlowScale(const T* flow, T* destination,
			const size_t& input_width, const size_t& input_height,
			const size_t& dst_width, const size_t& dst_height)
		{

			DownScaleGaussianGrayScale<T>(img, input_width, input_height,
				dst_width, dst_height, destination);

		}
	}
}