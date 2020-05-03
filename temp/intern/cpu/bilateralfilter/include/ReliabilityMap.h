#pragma once
#include<memory>
#include<stdint.h>
#include"FlowField.h"
#include"ImageRGB.h"

class ReliabilityMap
{
public:
	ReliabilityMap(const FlowField& flow, const ImageRGB& templateFrame, const ImageRGB& nextFrame);
	~ReliabilityMap();

	float GetReliability(uint32_t x, uint32_t y);
};

