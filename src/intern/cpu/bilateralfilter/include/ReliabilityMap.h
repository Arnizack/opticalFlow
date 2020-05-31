#pragma once
#include<memory>
#include<stdint.h>
#include"FlowField.h"
#include"ImageRGB.h"

namespace  cpu::bilateralfilter
{ 
	class ReliabilityMap
	{
	public:
		//RegionSize = n for the n x n Region that should considered for computing the reliability-value of an Pixel 
		ReliabilityMap(const core::FlowField& flow, const core::ImageRGB& templateFrame, const core::ImageRGB& nextFrame, uint8_t RegionSize);
		~ReliabilityMap();

		float GetReliability(uint32_t x, uint32_t y) const;

	private:
		std::vector<std::vector<float>> ReliabilityValues;
	};
}
