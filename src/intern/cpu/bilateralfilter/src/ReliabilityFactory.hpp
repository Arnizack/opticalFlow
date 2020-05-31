#pragma once
#include"ReliabilityMap.h"
#include<algorithm>
namespace cpu::bilateralfilter
{
	class ReliabilityFactory
	{
	public:
		ReliabilityFactory(uint8_t filterSize);
		float ReliabilityAt(uint32_t x,uint32_t y,const core::FlowField& flow, const core::ImageRGB& templateFrame, const core::ImageRGB& nextFrame);
	private:
		float FilterSize;
		float Error(const uint32_t x,const uint32_t y,const core::FlowVector& vec, const core::ImageRGB& templateFrame, const core::ImageRGB& nextFrame);
};

}
                