#pragma once
#include"CoordinateConverter.hpp"
#include<math.h>

namespace kdtree
{
	float CdfApprox(float x);

	uint8_t Floor(float x);

	float DistanceSquared(KdTreeVal& a, KdTreeVal& b);
	
}