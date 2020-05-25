#pragma once
#include<stdint.h>

namespace kdtree
{
	struct KdTreeVal
	{
		float X;
		float Y;
		float R;
		float G;
		float B;
		uint64_t mortonCode;
	public:
		KdTreeVal(float x, float y, float r, float g, float b);
		KdTreeVal();

		float& operator[](int index);
	};



	struct StandardVal
	{
		uint32_t X;
		uint32_t Y;
		unsigned char R;
		unsigned char G;
		unsigned char B;
		StandardVal(uint32_t x, uint32_t y, unsigned char r, unsigned char g, unsigned char b);
		StandardVal();
	};

}