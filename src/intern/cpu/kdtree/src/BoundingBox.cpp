
#pragma once
#include"BoundingBox.hpp"

void BoundingBox::enlarge(const BoundingBox & otherBoundingBox)
{
	for (int i = 0; i<5;i++)
	{
		if (Max[i] < otherBoundingBox.Max[i])
			Max[i] = otherBoundingBox.Max[i];
		if (Min[i] > otherBoundingBox.Min[i])
			Min[i] = otherBoundingBox.Min[i];
	}
}


