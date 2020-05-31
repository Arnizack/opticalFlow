#pragma once
#include"Color.h"
#include"FlowVector.h"
namespace core
{
	float ColorSqueredDistance(const Color& a, const Color& b);
	float Length(const FlowVector& a);
	float Length(int x, int y);

	float Distance(int x1, int y1, int x2, int y2);
}
                