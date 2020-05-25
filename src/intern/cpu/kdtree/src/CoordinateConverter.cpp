#include"CoordinateConverter.hpp"
#include<numeric>
#include<array>
#include<algorithm>
#include <stdexcept>

namespace kdtree
{
MortonCodeVal::MortonCodeVal(uint64_t value)
{
	Value = value;
}

MortonCodeVal::MortonCodeVal()
{
	Value = 0;
}

uint64_t CoordinateConverter::expandBits(uint64_t v)
{
	/*
	----- ---- --- --- --- --- --- --- --- --- --- --- --- --9 876 543 210
	----- ---- --- 98- --- --- --- --- --- --- --- --- --- --- -76 543 210
	----- ---- --- 98- --- --- --- --- --- 765 4-- --- --- --- --- --3 210
	----- ---- --- 98- --- --- -76 --- --- --5 4-- --- --- 32- --- --- -10
	----- ---9 --- -8- --- 7-- --6 --- -5- --- 4-- --3 --- -2- --- 1-- --0
	*/
	v = (v * 0x100000001u) & 0x300000000ffu;
	v = (v * 0x10001u) & 0x30000f000ffu;
	v = (v * 0x101u) & 0x300c0300c03u;
	v = (v * 0x11u) & 0x210842108421u;
	return v;
}

uint64_t CoordinateConverter::shrinkBits(uint64_t v)
{
	/*
	------9----8----7----6----5----4----3----2----1----0
		  1000010000100001000010000100001000010000100001 = 0x210842108421u
	----------98--------76--------54--------32--------10
			  110000000011000000001100000000110000000011 = 0x300c0300c03u
	----------98----------------7654----------------3210
			  110000000000000000111100000000000000001111 = 0x30000f0000fu
	----------98--------------------------------76543210
			  110000000000000000000000000000000011111111 = 0x300000000ffu
	------------------------------------------9876543210
											  1111111111 = 0x3ffu
	*/
	v &= 0x210842108421u;
	v = (v | (v >> 4)) & 0x300c0300c03u;
	v = (v | (v >> 8)) & 0x30000f0000fu;
	v = (v | (v >> 16)) & 0x300000000ffu;
	v = (v | (v >> 32)) & 0x3ffu;


	return v;


}

CoordinateConverter::CoordinateConverter(StandardVal maxPoint, float sigma_distance, float sigma_color)
{
	
	Sigma_color = sigma_color;
	Sigma_distance = sigma_distance;


	MaxPoint = StandardToKdTree(maxPoint);

	KdTreeVal MaxPointForLambda = MaxPoint;

	//create indices
	std::array<unsigned short, 5> indices;
	std::iota(indices.begin(), indices.end(), 0);

	/*
	std::stable_sort(indices.begin(), indices.end(),
		[&MaxPointForLambda](short a, short b) {return MaxPointForLambda[a] > MaxPointForLambda[b]; });
	*/
	DimmensionOrder = indices;

	

}

MortonCodeVal CoordinateConverter::StandardToMortonCode(StandardVal value)
{
	return KdTreeToMortonCode(StandardToKdTree(value));
}

KdTreeVal CoordinateConverter::MortonCodeToKdTree(MortonCodeVal value)
{
	KdTreeVal kdTreeValue(0, 0, 0, 0, 0);
	float maxValue = MaxPoint[DimmensionOrder[0]];
	for (int i = 0; i < 5; i++)
	{
		uint64_t shiffted = value.Value >> ((4-i) + 14);
		uint32_t encodedValue = static_cast<uint32_t>(shrinkBits(shiffted));
		if (encodedValue == 1023)
			encodedValue = 1024;
		float kdVal = static_cast<float>( ((double) (encodedValue) / 1024.0)*maxValue );
		kdTreeValue[DimmensionOrder[i]] = kdVal;

	}
	return kdTreeValue;
}



// Calculates a 50-bit Morton code for the
// given 5D point located within the unit cube [0,1].
uint64_t CoordinateConverter::createMortonCode(std::array<float,5> point)
{
	uint64_t mortonCode = 0;
	uint32_t x;
	uint64_t xExpanded;
	for (uint32_t i = 0; i < 5; i++)
	{
		x = std::min(std::max(point[4-i] * 1024.0f, 0.0f), 1023.0f);
		xExpanded = expandBits(x);
		mortonCode += xExpanded << (i+14);
	}
	return mortonCode;

}

StandardVal CoordinateConverter::KdTreeToStandard(KdTreeVal value)
{
	uint32_t x = static_cast<uint32_t> (value.X*Sigma_distance);
	uint32_t y = static_cast<uint32_t> (value.Y*Sigma_distance);
	unsigned char r = static_cast<unsigned char> (value.R * Sigma_color);
	unsigned char g = static_cast<unsigned char> (value.G * Sigma_color);
	unsigned char b = static_cast<unsigned char> (value.B * Sigma_color);

	return StandardVal(x, y, r, g, b);
}

MortonCodeVal CoordinateConverter::KdTreeToMortonCode(KdTreeVal value)
{
	std::array<float, 5> point;
	float maxValue = MaxPoint[DimmensionOrder[0]];
	for (int i = 0; i < 5; i++)
	{
		point[i] = value[DimmensionOrder[i]] / maxValue;
	}
	return createMortonCode(point);
}

StandardVal CoordinateConverter::MortonCodeToStandard(MortonCodeVal value)
{
	return KdTreeToStandard(MortonCodeToKdTree(value));
}

KdTreeVal CoordinateConverter::StandardToKdTree(StandardVal value)
{
	float x = value.X / Sigma_distance;
	float y = value.Y / Sigma_distance;
	float r = value.R / Sigma_color;
	float g = value.G / Sigma_color;
	float b = value.B / Sigma_color;
	
	KdTreeVal kdVal(x, y, r, g, b);
	return kdVal;
}

}