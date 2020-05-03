#pragma once

#include"ImageRGB.h"
#include<stdint.h>
#include<array>

class ImageRGBGradient
{
public:
	ImageRGBGradient(uint32_t width, uint32_t height);
	
	~ImageRGBGradient();

	std::array<float, 2> getGradient(uint32_t x, uint32_t y) const;

	void setGradient(uint32_t x, uint32_t, std::array<float, 2> gradient);



};