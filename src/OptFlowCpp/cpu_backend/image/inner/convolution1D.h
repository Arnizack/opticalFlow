#pragma once
#include<algorithm>
#include"ArrayHelper.h"

namespace cpu_backend
{

	enum class Direction
	{
		X,Y
	};



	namespace _inner
	{
		

		template<Direction direction>
		inline int CorrespondingX(const int& kernel_idx, const int& kernel_length_half, const int& x)
		{
			switch (direction)
			{
			case Direction::X:
				return x - kernel_length_half + kernel_idx;
			default:
				return x;
			}
		}

		template<Direction direction>
		inline int CorrespondingY(const int& kernel_idx, const int& kernel_length_half, const int& y)
		{
			switch (direction)
			{
			case Direction::Y:
				return y - kernel_length_half + kernel_idx;
			default:
				return y;
			}
		}




	}

	template<class T, Padding padding, Direction direction>
	void Convolute1D(T* img, int width, int height,
		T* kernel, int kernel_length, T* destination)
	{
		int kernel_length_half = kernel_length / 2;
		#pragma omp parallel for
		for(int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float sum = 0;
				for (int kernel_idx = 0; kernel_idx < kernel_length; kernel_idx++)
				{
					int x1 = _inner::CorrespondingX<direction>(kernel_idx, kernel_length_half, x);
					int y1 = _inner::CorrespondingY<direction>(kernel_idx, kernel_length_half, y);
					T img_val = _inner::GetValueAt<T,padding>(x1, y1, width, height, img);
					sum += img_val * kernel[kernel_idx];
				}
				destination[width * y + x] = sum;
			}
		}
	}

	
}