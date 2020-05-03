#pragma once
#include"GradientBuilder.h"
#include<array>
#include"CudaImageRGBGradient.h"
#include"CUDAImageRGB.h"

class CudaGradientBuilder
{
public:
	CudaGradientBuilder(std::array<float, 5> kernel, PaddingMode mode = PaddingMode::WEIGHTED);

	CudaImageRGBGradient differentiate(const CudaImageRGB& image);
};

