#include<gtest/gtest.h>
#include"CoordinateConverter.hpp"

using namespace kdtree;

TEST(KDTree, StandardToMortonCode)
{
	
	CoordinateConverter converter(StandardVal(1920,1080,(unsigned char)223, (unsigned char)222, (unsigned char)224),5,8 );
	MortonCodeVal mortonCode = converter.StandardToMortonCode(StandardVal(1920, 1080, (unsigned char)223, (unsigned char)222, (unsigned char)224));
	/*
	calculation
	Input: 1920 1080 223 222 224
	To KDTree:
	1920/5 = 384; 1080/5 = 216; 223/8 = 27.875; 222/8 = 27.75; 224/8 = 28
	Dimmensions: 0|1|2|3|4
	Order:       0|1|4|3|2
	To [0,1] Unit Cube

	384/384 = 1; 216/384 = 0.5625; 28/384 = 0.07291; 27.875/384 = 0.07259; 27.75/384 = 0.07227
	To [0,1023] Unit Cube
	1*1024 -> 1023 (verringert); 0.5625*1024 = 576; 0.07291*1024 = 74; 0.07259*1024 = 74; 0.07227*1024 = 74

	To Bit
	1023 = 1111111111
	576  = 1001000000
	74   = 0001001010
	74   = 0001001010
	74   = 0001001010

	Morton Code:
	1    1    1    1    1    1    1    1    1    1
	 1    0    0    1    0    0    0    0    0    0
	  0    0    0    1    0    0    1    0    1    0
	   0    0    0    1    0    0    1    0    1    0
	    0    0    0    1    0    0    1    0    1    0
	11000100001000011111100001000010111100001011110000

	To 64 Bit: Shift um 14 Bit
	11000100001000011111100001000010111100001011110000              
	0000000000000000000000000000000000000000000000000000000000000000
	->
	1100010000100001111110000100001011110000101111000000000000000000 = 0xc421f842f0bc0000u
	=14132850072055709696

	*/
	EXPECT_EQ(mortonCode.Value, 0xc421f842f0bc0000u);
}

TEST(KDTree, MortonCodeToStandard)
{
	StandardVal input(1920, 1080, (unsigned char)223, (unsigned char)222, (unsigned char)224);
	CoordinateConverter converter(input, 5, 8);
	MortonCodeVal mortonCode = converter.StandardToMortonCode(input);
	StandardVal output = converter.MortonCodeToStandard(mortonCode);
	EXPECT_NEAR((int)output.X, (int)input.X,2);
	EXPECT_NEAR((int)output.Y, (int)input.Y,2);
	EXPECT_NEAR((int)output.R, (int)input.R,2);
	EXPECT_NEAR((int)output.G, (int)input.G,2);
	EXPECT_NEAR((int)output.B, (int)input.B,2);

}

TEST(KDTree, StandardToMortonCode2)
{
	CoordinateConverter converter(StandardVal(721, 681, (unsigned char)225, (unsigned char)225, (unsigned char)225),20,225);
	MortonCodeVal mortonCode = converter.StandardToMortonCode(StandardVal(0, 0, (unsigned char)225, (unsigned char)225, (unsigned char)225));
	EXPECT_EQ(2, 3);
}