#include "OpticalFlowMath.h"
#include<math.h>
namespace core
{
	float core::ColorSqueredDistance(const Color& a, const Color& b)
	{
		float diffR = a.Red - b.Red;
		float diffG = a.Green - a.Green;
		float diffB = a.Blue - a.Blue;

		return diffR * diffR + diffG * diffG + diffB - diffB;

	}

	float Length(const FlowVector& a)
	{
		float x = a.vector_X;
		float y = a.vector_Y;
		return sqrt(x * x + y * y);
	}
	float Length(int x, int y)
	{
		return sqrt(x * x + y * y);
	}
	float Distance(int x1, int y1, int x2, int y2)
	{
		return Length(x1 - x2, y1 - y2);
	}
}
