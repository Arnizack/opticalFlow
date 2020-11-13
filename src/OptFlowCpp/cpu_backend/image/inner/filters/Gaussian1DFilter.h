#pragma once
#include"../convolution1D.h"
#include<vector>

#include <cmath>

#define M_PI 3.14159265358979323846

namespace cpu_backend
{

	namespace _inner
	{
		template<class T>
		std::vector<T> CreateKernel(long double standard_deviation)
		{

			size_t kernel_length = ceil(standard_deviation) * 6 + 1; 
			std::vector<T> kernel(kernel_length);
			
			const long double sqrt2Pi = sqrt((long double)2.0 * M_PI);

			//T scaler = 1.0 / (standard_deviation * sqrt2Pi);

			//T x_scaler = 1.0 / (2 * standard_deviation * standard_deviation);

			long double scaler = (standard_deviation * sqrt2Pi);

			long double x_scaler = (2 * standard_deviation * standard_deviation);

			for (int i = 0; i < kernel_length; i++)
			{
				long double x = (long double)i - floor(kernel_length / 2);
				kernel[i] =  exp(-(x * x) / x_scaler) / scaler;
			}

			return kernel;
		}
	
		template<class T, Direction _Direction, Padding _Padding>
		void Gaussian1DFilter(T* image, int width, int height, double standard_deviation, T* destination)
		{
			std::vector<T> kernel = _inner::CreateKernel<T>(standard_deviation);
			Convolute1D<T, _Padding, _Direction>(image, width, height, kernel.data(), kernel.size(), destination);

		}
	}

}