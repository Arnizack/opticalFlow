#include<gtest/gtest.h>
#include"ColorSpaceConverter.hpp"
#include"FlowField.h"
#include"ImageRGB.h"

namespace visualization
{

	TEST(visualization, HSVToRGB)
	{
		ColorSpaceConverter converter;

		std::vector<HSV> inputs =
		{ {0,1,1},
			{100,0.50,0.60}
		};
		std::vector<core::Color> expected =
		{
			{255,0,0},
			{102,153,76}
		};
		for (int i = 0; i < inputs.size(); i++)
		{
			core::Color actualColRBG = converter.HSVToRGB(inputs[i]);
			core::Color expectedColRGB = expected[i];
			EXPECT_NEAR(actualColRBG.Red, expectedColRGB.Red,0.01);
			EXPECT_NEAR(actualColRBG.Green, expectedColRGB.Green, 0.01);
			EXPECT_NEAR(actualColRBG.Blue, expectedColRGB.Blue, 0.01);
		}

		
	}


	TEST(visualization, HSVToRGB2)
	{
		ColorSpaceConverter converter;
		int width = 1000;
		int heigth = 1000;
		core::ImageRGB img(width,heigth);
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < heigth; y++)
			{
				float xval = sin(((float)y )/ 100) * 180+180;
				core::Color color = converter.HSVToRGB(HSV(xval, 1, 1));
				img.SetPixel(x, y, color);
			}
		}

		
		img.Save("testHue.png");
	}
}