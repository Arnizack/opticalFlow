
#pragma once
#include<stdint.h>
#include"BoundingBox.hpp"
class MortonNode
{
public:
	uint32_t Split;
	uint32_t First;
	uint32_t Last;
	
	MortonNode* Parent = nullptr;

	bool AlreadyWorkedOn = false;
	BoundingBox Bounds;
};