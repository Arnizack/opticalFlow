#include "ReliabilityMap.h"
#include"ReliabilityFactory.hpp"
#include<cmath>
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
		Width = width;
		Heigth = heigth;

	}
	ReliabilityMap::~ReliabilityMap()
	{
	}
	float ReliabilityMap::GetReliability(uint32_t x, uint32_t y) const
	{
		//return 1;
		return ReliabilityValues[x][y];
	}
	core::ImageRGB ReliabilityMap::toImage()
	{
		core::ImageRGB result(Width,Heigth);

		for (uint32_t x = 0; x < Width; x++)
		{
			for (uint32_t y = 0; y < Heigth; y++)
			{
				float reliablitiy = log(GetReliability(x, y));
				result.SetPixel(x, y, core::Color(reliablitiy,reliablitiy,reliablitiy));
			}
		}
		return result;
	}

}