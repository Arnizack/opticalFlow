#include "ImageRGBGradient.h"
namespace cpu::imagegradient
{
	ImageRGBGradient::ImageRGBGradient(uint32_t width, uint32_t height)
	{
	}
	ImageRGBGradient::~ImageRGBGradient()
	{
	}
	std::array<float, 2> ImageRGBGradient::getGradient(uint32_t x, uint32_t y) const
	{
		return std::array<float, 2>();
	}
	void ImageRGBGradient::setGradient(uint32_t x, uint32_t, std::array<float, 2> gradient)
	{
	}
}