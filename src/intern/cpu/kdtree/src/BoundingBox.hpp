#pragma once
#include<array>
#include<CoordinateConverter.hpp>
#include <KDTreeSpaces.h>

class BoundingBox
{
public:
	std::array<float, 5> Max;
	std::array<float, 5> Min;

	//updates the BoundingBox, so that the given boundingbox fits inside
	void enlarge(const BoundingBox& otherBoundingBox);
	void enlarge(const kdtree::KdTreeVal & otherValue);
};