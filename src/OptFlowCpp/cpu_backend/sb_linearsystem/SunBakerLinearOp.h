#pragma once
#include"core/linalg/ILinearOperator.h"
#include"core/IArray.h"
#include<memory>
#include"../Array.h"

namespace cpu_backend
{

	class SunBakerLinearOp final: public core::ILinearOperator<
		std::shared_ptr<core::IArray<double, 1>>,
		std::shared_ptr<core::IArray<double, 1>>>
	{
		using PtrGrayImg = std::shared_ptr<core::IArray<double, 1 >> ;
		/*
		Matrix representation:
			    |K+A C  |
			A = |C   K+B|

			Shape of A (width*height*2,width*height*2)

			with:
				K * x = - 2 lambda_kernel * mat(ker(x))
						 | 1   2 1 |
				kernel = | 2 -12 2 | / 12
						 | 1   2 1 |
				
				ker(x) uses width, height

				C = diag_matrix(c_diags)
				A = diag_matrix(a_diags)
				B = diag_matrix(b_diags)

				

		*/
	public:
		SunBakerLinearOp(
			size_t width,
			size_t height,
			std::shared_ptr<Array<double, 1>> a_diags,
			std::shared_ptr<Array<double, 1>> b_diags,
			std::shared_ptr<Array<double, 1>> c_diags,
			double lambda_kernel);
		
		virtual PtrGrayImg Apply(const PtrGrayImg vec) override;
		virtual void ApplyTo(PtrGrayImg dst, const PtrGrayImg vec) override;
		virtual std::shared_ptr<core::ILinearOperator<PtrGrayImg, PtrGrayImg>> Transpose() override;
		virtual bool IsSymetric() override;

	private:
		size_t _width;
		size_t _height;
		
		std::shared_ptr<Array<double, 1>> _a_diags;
		std::shared_ptr<Array<double, 1>> _b_diags;
		std::shared_ptr<Array<double, 1>> _c_diags;

		double _lambda_kernel;

		


	
	};
}