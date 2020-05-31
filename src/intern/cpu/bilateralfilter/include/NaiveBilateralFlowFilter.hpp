#pragma once
#include"ImageRGB.h"
#include"FlowField.h"
#include"ReliabilityMap.h"
namespace cpu::bilateralfilter
{
	class NaiveBilateralFilter
	{
	//this is the more accurate implementation
	public:
		core::FlowField filter(cpu::bilateralfilter::ReliabilityMap& map,
			core::FlowField& flow,core::ImageRGB& templateImg,float delta_d, float delta_c);
	};

}
                