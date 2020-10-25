#pragma once
#include<memory>
#include"../../Array.h"
#include"convolution1D.h"

namespace cpu_backend
{

	template<class T>
	class DerivativeCalculator
	{
	public:
		void ComputeDerivativeX(T* img, int width, int height,T* dst)
		{
			Convolute1D<T, Padding::SYMMETRIC, Direction::X>(img, width, height, kernel.data(), 5, dst);
		}
		void ComputeDerivativeY(T* img, int width, int height,T* dst)
		{
			Convolute1D<T, Padding::SYMMETRIC, Direction::Y>(img, width, height, kernel.data(), 5, dst);
		}

		// div img = d/dx img + d/dy img
		void Divergence(T* img, int width, int height, T* dst)
		{
			int size = width * height;
			T* temp = malloc(size * sizeof(T));
			ComputeDerivativeX(img, width, height, temp);
			ComputeDerivativeY(img, width, height, dst);
			for (int i = 0; i < size; i++)
				dst[i] += temp[i];

		}
	private:
		//kernel = [-1, 8, 0, -8, 1]/12
		//std::array<float, 5> kernel_x = 
		//{ -1.0 / 12.0, 8.0 / 12.0,0,-8.0 / 12.0,1.0 / 12.0 };

		std::array<float,5> kernel = 
		{ 1.0 / 12.0, -8.0 / 12.0,0,8.0 / 12.0, -1.0 / 12.0 };
	};
}