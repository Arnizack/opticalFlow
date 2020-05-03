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
		ReliabilityMap(const core::FlowField& flow, const core::ImageRGB& templateFrame, const core::ImageRGB& nextFrame);
		~ReliabilityMap();

		float GetReliability(uint32_t x, uint32_t y);
	};
}
