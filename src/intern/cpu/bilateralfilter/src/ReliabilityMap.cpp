#include "ReliabilityMap.h"
#include"ReliabilityFactory.hpp"
namespace cpu::bilateralfilter
{
	ReliabilityMap::ReliabilityMap(const core::FlowField & flow, const core::ImageRGB & templateFrame, const core::ImageRGB & nextFrame, uint8_t RegionSize)
	{
		ReliabilityFactory factory(RegionSize);
		uint32_t width = templateFrame.GetWidth();
		uint32_t heigth = templateFrame.GetHeight();

		ReliabilityValues = std::vector<std::vector<float>>(width, std::vector<float>(heigth));
		
		for(uint32_t x = 0; x < width; x++)
		{
			for (uint32_t y = 0; y < heigth; y++)
			{
				ReliabilityValues[x][y] = factory.ReliabilityAt(x, y, flow, templateFrame, nextFrame);
			}
		
		}

	}
	ReliabilityMap::~ReliabilityMap()
	{
	}
	float ReliabilityMap::GetReliability(uint32_t x, uint32_t y) const
	{
		return ReliabilityValues[x][y];
	}
}