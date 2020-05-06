#include<gtest/gtest.h>
#include<ImageRGB.h>
#include"loggerHelper.hpp"

TEST(coreImageRGB, ImageRGBloadCalib1)
{
	
	std::string filepath = __TESTDATADIR__;
	filepath += "\\CalibrationPicture.png";
	core::ImageRGB img(filepath);

	std::vector<std::array<int, 5>> values = { 
		{0,0,255,255,255},
	{1,0,157,157,157},
	{0,1,58,58,58},
	{1,1,73,73,73},
	{0,2,2,2,2},
	{1,2,29,29,29} };

	EXPECT_EQ(3, img.GetHeight());
	EXPECT_EQ(2, img.GetWidth());

	for(auto val : values)
	{

		auto actual_col1 = img.GetPixel(val[0], val[1]);
	
		int actual_red = (int)actual_col1.Red;
		int actual_green = (int)actual_col1.Green;
		int actual_blue = (int)actual_col1.Blue;

		EXPECT_EQ(actual_red, (int)val[2]);

		EXPECT_EQ(actual_green, (int)val[3]);
		EXPECT_EQ(actual_blue, (int)val[4]);
	}
}

TEST(coreImageRGB, ImageRGBloadCalib2)
{
	std::string filepath = __TESTDATADIR__;
	filepath += "\\CalibrationPicture2.png";
	core::ImageRGB img(filepath);
	

	std::vector<std::array<int, 5>> values = {
		{0,0,254,254,254},
		{1,0,0,0,0},
		{0,1,152,152,152},
		{1,1,103,103,103}

	};

	EXPECT_EQ(2, img.GetHeight());
	EXPECT_EQ(2, img.GetWidth());

	for (auto val : values)
	{

		auto actual_col1 = img.GetPixel(val[0], val[1]);

		int actual_red = (int)actual_col1.Red;
		int actual_green = (int)actual_col1.Green;
		int actual_blue = (int)actual_col1.Blue;

		EXPECT_EQ(actual_red, (int)val[2]);

		EXPECT_EQ(actual_green, (int)val[3]);
		EXPECT_EQ(actual_blue, (int)val[4]);
	}

}

TEST(coreImageRGB, toManyChannels)
{
	std::string filepath = __TESTDATADIR__;
	filepath += "\\toManyChannels.png";
	core::ImageRGB img(filepath);

	EXPECT_EQ(-1, img.GetHeight());
	EXPECT_EQ(-1, img.GetWidth());
}

TEST(coreImageRGB, JpgColor)
{
	std::string filepath = __TESTDATADIR__;
	filepath += "\\colorTest.jpg";
	core::ImageRGB img(filepath);

	EXPECT_EQ(8, img.GetHeight());
	EXPECT_EQ(8, img.GetWidth());

	auto actual_col1 = img.GetPixel(4, 4);

	int actual_red = (int)actual_col1.Red;
	int actual_green = (int)actual_col1.Green;
	int actual_blue = (int)actual_col1.Blue;

	EXPECT_EQ(actual_red, 177);

	EXPECT_EQ(actual_green, 94);
	EXPECT_EQ(actual_blue, 110);
}
TEST(coreImageRGB, SaveImage)
{
	core::ImageRGB img(2,3);
	std::string filepath = __TESTDATADIR__;
	filepath += "\\testSave.png";
	core::Color color;
	color.Red = 255;
	color.Green = 120;
	color.Blue = 100;
	img.SetPixel(1, 1, color);
	img.save(filepath);

	core::ImageRGB imgLoaded(filepath);
	core::Color actual_color = imgLoaded.GetPixel(1, 1);

	EXPECT_EQ(actual_color.Red, color.Red);

	EXPECT_EQ(actual_color.Green, color.Green);
	EXPECT_EQ(actual_color.Blue, color.Blue);

}