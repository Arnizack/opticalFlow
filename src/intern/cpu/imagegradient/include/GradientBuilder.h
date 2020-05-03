#pragma once

#include"ImageRGB.h"
#include"ImageRGBGradient.h"
#include<array>

namespace cpu::imagegradient
{

	enum class PaddingMode
	{
		WEIGHTED,
		ZEROS
	};

	class GradientBuilder
	{
	public:
		GradientBuilder(std::array<float, 5> kernel, PaddingMode mode = PaddingMode::WEIGHTED);

		ImageRGBGradient differentiate(const core::ImageRGB& image);

	};

}