#include "ReliabilityMap.h"
namespace cpu::bilateralfilter
{
	ReliabilityMap::ReliabilityMap(const core::FlowField & flow, const core::ImageRGB & templateFrame, const core::ImageRGB & nextFrame)
	{
	}
	ReliabilityMap::~ReliabilityMap()
	{
	}
	float ReliabilityMap::GetReliability(uint32_t x, uint32_t y)
	{
		return 0.0f;
	}
}