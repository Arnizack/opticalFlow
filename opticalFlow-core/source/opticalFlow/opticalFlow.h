#pragma once
#include<stdint.h>

namespace opticalFlow
{
	//TODO add Mask
	float* calcOpticalFlow(float* const pixels, uint32_t width, uint32_t heigth);
}