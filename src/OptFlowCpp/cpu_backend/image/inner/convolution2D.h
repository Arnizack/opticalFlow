#pragma once
#include"ArrayHelper.h"

namespace cpu_backend
{

	

	template<class T, size_t kernel_width, 
		size_t kernel_height>
	constexpr inline T Convolute2DAt(const int& x, const int& y,const T* image,
		const size_t& width,const size_t& height, const T* kernel
		
		)
	{
		const int kernel_width_half = kernel_width / 2;
		const int kernel_height_half = kernel_height / 2;
		T sum = 0;
		
		for (size_t kernel_idx_y = 0; 
			kernel_idx_y < kernel_height; 
			kernel_idx_y++)
		{
			int y_remap = y + kernel_idx_y - kernel_height_half;

			for (size_t kernel_idx_x = 0;
				kernel_idx_x < kernel_width;
				kernel_idx_x++)
			{
				int x_remap = x + kernel_idx_x - kernel_width_half;
				T kernel_val = kernel[kernel_idx_y * kernel_width + kernel_idx_x];
				T img_value = _inner::GetValueAt<T,Padding::ZEROS>(x_remap, y_remap, width, height, image);
				sum += kernel_val * img_value;
			}
		}
		
		return sum;
		
	}

	template<class T, size_t kernel_width,
		size_t kernel_height>
		inline void Convolute2D(const T* image, T* destination, size_t width, size_t height, 
			T* kernel)
	{

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < height; x++)
			{
				double sum = Convolute2DAt<T, kernel_width, kernel_height, kernel>
					(x, y, image, width, height);
				destination[width * y + x] = sum;
			}
		}
	}
}