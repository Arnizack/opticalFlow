#include"KDTreeSpaces.h"
#include <stdexcept>

namespace kdtree
{
	

	KdTreeVal::KdTreeVal(float x, float y, float r, float g, float b)
	{
		X = x;
		Y = y;
		R = r;
		G = g;
		B = b;
	}
	KdTreeVal::KdTreeVal()
	{
		X = 0;
		Y = 0;
		R = 0;
		G = 0;
		B = 0;
	}

	float& KdTreeVal::operator[](int index)
	{
		switch (index)
		{
		case 0:
			return X;
		case 1:
			return Y;
		case 2:
			return R;
		case 3:
			return G;
		case 4:
			return B;
		default:
			throw std::out_of_range("Index is out range in [0,4]");
			break;
		}
	}



	StandardVal::StandardVal(uint32_t x, uint32_t y, unsigned char r, unsigned char g, unsigned char b)
	{
		X = x;
		Y = y;
		R = r;
		G = g;
		B = b;
	}
	StandardVal::StandardVal()
	{
		X = 0;
		Y = 0;
		R = (unsigned char)0;
		G = (unsigned char)0;
		B = (unsigned char)0;
	}

}