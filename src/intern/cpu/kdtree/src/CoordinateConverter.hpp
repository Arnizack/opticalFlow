#pragma once
#include<stdint.h>
#include<array>
#include"KDTreeSpaces.h"

namespace kdtree
{
	struct MortonCodeVal
	{
		uint64_t Value;
		MortonCodeVal(uint64_t value); 
		MortonCodeVal();
	};



	class CoordinateConverter
	{
		/*
		There are three spaces
		The MortonCodeSpace
		The KdTreeSpace
		The StandardSpace

		*/

	public:
		//The maxPoint shoud have the max Values of the Dimensions
		CoordinateConverter(StandardVal maxPoint, float sigma_distance, float sigma_color);

		MortonCodeVal StandardToMortonCode(StandardVal value);

		KdTreeVal MortonCodeToKdTree(MortonCodeVal value);

		StandardVal KdTreeToStandard(KdTreeVal value);

		MortonCodeVal KdTreeToMortonCode(KdTreeVal value);

		StandardVal MortonCodeToStandard(MortonCodeVal value);

		KdTreeVal StandardToKdTree(StandardVal value);

		uint64_t shrinkBits(uint64_t v);
		uint64_t expandBits(uint64_t v);

	private:
		uint64_t createMortonCode(std::array<float, 5> point);

		KdTreeVal MaxPoint = KdTreeVal(0,0,0,0,0);
		float Sigma_color;
		float Sigma_distance;

		std::array<unsigned short, 5> DimmensionOrder;

	
	};
}