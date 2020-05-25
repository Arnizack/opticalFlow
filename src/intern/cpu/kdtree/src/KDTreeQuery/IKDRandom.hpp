#pragma once
#include<stdint.h>
namespace kdtree
{
	class IKDRandom
	{
	public:
		//a uniform random Variable
		//pointX is the X Coordinate of the input in the KD Query
		//pointY is the Y Coordinate of the input in the KD Query
		virtual float Urand(uint32_t nodeIdx, uint32_t pointX,uint32_t pointY) = 0;
	};
}