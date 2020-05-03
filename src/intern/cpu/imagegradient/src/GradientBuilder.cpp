#include "GradientBuilder.h"
namespace cpu::imagegradient
{
	GradientBuilder::GradientBuilder(std::array<float, 5> kernel, PaddingMode mode)
	{
	}
	ImageRGBGradient GradientBuilder::differentiate(const core::ImageRGB & image)
	{
		return ImageRGBGradient(0, 0);
	}
}