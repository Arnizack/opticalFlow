#pragma once
#include"pch.h"
#include "DerivativeCalculator.h"
#include"convolution1D.h"

namespace cpu_backend
{
	void DerivativeCalculator::ComputeDerivativeX(float* img, int width, int height, float* dst)
	{
		Convolute1D<float, Padding::REPLICATE, Direction::X>(img, width, height, kernel.data(), 5, dst);
	}
	void DerivativeCalculator::ComputeDerivativeY(float* img, int width, int height, float* dst)
	{
		Convolute1D<float, Padding::REPLICATE, Direction::X>(img, width, height, kernel.data(), 5, dst);
	}
}