#pragma once
namespace kdtree
{
	struct KDResult
	{
		uint32_t X;
		uint32_t Y;
		unsigned char R;
		unsigned char G;
		unsigned char B;

		float Weight;
		int Level;

		KDResult(uint32_t x, uint32_t y, unsigned char r, unsigned char g, unsigned char b, float weight, int level = 0);
		KDResult();
	};
}