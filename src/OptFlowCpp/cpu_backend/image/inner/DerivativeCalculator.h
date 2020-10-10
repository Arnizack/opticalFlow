#pragma once
#include<memory>
#include"../../Array.h"

namespace cpu_backend
{

	class DerivativeCalculator
	{
	public:
		void ComputeDerivativeX(float* img, int width, int height,float* dst);
		void ComputeDerivativeY(float* img, int width, int height,float* dst);
	private:
		//kernel = [-1, 8, 0, -8, 1]/12
		std::array<float, 5> kernel = { -1.0 / 12.0, 8.0 / 12.0,0,-8.0 / 12.0,1.0 / 12.0 };
	};
}