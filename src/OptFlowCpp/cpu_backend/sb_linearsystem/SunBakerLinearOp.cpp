#pragma once

#include"SunBakerLinearOp.h"
#include"../image/inner/convolution2D.h"
#include"../Array.h"
#include <omp.h>
#include"../Base.h"

namespace cpu_backend
{

	namespace _inner
	{
		constexpr inline double Convolute2DAt(const double* image,const int& dst_x,const int& dst_y,const int& width,
			const int& height,const double* kernel)
		{
			const int kernel_height = 3;
			const int kernel_width = 3;

			const int kernel_half_height = kernel_height / 2;
			const int kernel_half_width = kernel_width / 2;

			double sum = 0;

			for (int kernel_y = 0; kernel_y < kernel_height; kernel_y++)
			{
				int src_y = dst_y - kernel_half_height + kernel_y;
				if(src_y>=0 && src_y < height)
				{
					for (int kernel_x = 0; kernel_x < kernel_width; kernel_x++)
					{
						int src_x = dst_x - kernel_half_width + kernel_x;
						if (src_x >= 0 && src_x < width)
						{
							sum += image[src_y * width + src_x] * kernel[kernel_width * kernel_y + kernel_x];
						}
					}
				}
			}
			return sum;
		}
	}

	using PtrGrayImg = std::shared_ptr<core::IArray<double, 1>>;
	SunBakerLinearOp::SunBakerLinearOp(size_t width, size_t height, 
		std::shared_ptr<Array<double, 1>> a_diags, 
		std::shared_ptr<Array<double, 1>> b_diags, 
		std::shared_ptr<Array<double, 1>> c_diags, 
		double lambda_kernel)
		: _width(width), _height(height), _a_diags(a_diags), _b_diags(b_diags),
		_c_diags(c_diags),_lambda_kernel(lambda_kernel)
	{
	}
	PtrGrayImg SunBakerLinearOp::Apply(const PtrGrayImg vec)
	{

		auto result = std::make_shared< Array<double,1>>(vec->Shape);
		ApplyTo(result, vec);
		return result;
	}
	void SunBakerLinearOp::ApplyTo(PtrGrayImg dst, const PtrGrayImg vec)
	{
		//OPF_PROFILE_FUNCTION();
		double* __restrict dst_data = std::static_pointer_cast<
			Array<double, 1>>(dst)->Data();
		double* __restrict vec_data = std::static_pointer_cast<
			Array<double, 1>>(vec)->Data();

		/*
		vec = |upper_half|
			  |lower_half|
		*/

		double* vec_upper = vec_data;
		double* vec_lower = vec_data+_width*_height;

		double* __restrict dst_upper = dst_data;
		double* __restrict dst_lower = dst_data + _width * _height;

		double* __restrict ptr_a_diags = _a_diags->Data();
		double* __restrict ptr_b_diags = _b_diags->Data();
		double* __restrict ptr_c_diags = _c_diags->Data();

		
		const double _kernel[9] =
		{
			1.0 / 12.0, 2.0 / 12, 1.0 / 12,
			2.0 / 12.0, -1, 2.0 / 12.0,
			1.0 / 12.0, 2.0 / 12.0, 1.0 / 12
		};
		


		//dst_upper = A*vec_upper - 2 lambda_kernel * ker(vec_upper)
		//dst_upper += C*vec_lower
		#pragma omp parallel for collapse(2)
		for (int y = 0; y < _height; y++)
		{
			for (int x = 0; x < _width; x++)
			{
				int coord = y * _width + x;
				double result = ptr_a_diags[coord] * vec_upper[coord];
				//double kernel_result =
				//	Convolute2DAt<double,3,3>(x,y,vec_upper, _width, _height, _kernel);
				double kernel_result =
					_inner::Convolute2DAt(vec_upper, x,y, _width, _height, _kernel);

				result -= 2 * _lambda_kernel * kernel_result;
				result += ptr_c_diags[coord] * vec_lower[coord];

				dst_upper[coord] = result;


			}
		}
		
		//dst_upper = B*vec_lower - 2 lambda_kernel * ker(vec_lower)
		//dst_upper += C*vec_upper
		#pragma omp parallel for collapse(2)
		for (int y = 0; y < _height; y++)
		{
			for (int x = 0; x < _width; x++)
			{
				int coord = y * _width + x;
				double result = ptr_b_diags[coord] * vec_lower[coord];
				//double kernel_result =
				//	Convolute2DAt<double, 3, 3>(x, y, vec_lower, _width, _height, _kernel);
				double kernel_result =
					_inner::Convolute2DAt(vec_lower, x, y, _width, _height, _kernel);

				result -= 2 * _lambda_kernel * kernel_result;
				result += ptr_c_diags[coord] * vec_upper[coord];

				dst_lower[coord] = result;
			}
		}


	}

	std::shared_ptr<core::ILinearOperator<PtrGrayImg, PtrGrayImg>> SunBakerLinearOp::Transpose()
	{
		auto linop = std::make_shared<SunBakerLinearOp>(_width, _height, _a_diags, _b_diags, _c_diags, _lambda_kernel);
		return std::static_pointer_cast<core::ILinearOperator<PtrGrayImg, PtrGrayImg>,
			SunBakerLinearOp>(linop);
	}

	bool SunBakerLinearOp::IsSymetric()
	{
		return true;
	}

}