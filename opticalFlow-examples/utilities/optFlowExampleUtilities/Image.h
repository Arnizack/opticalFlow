#pragma once
#include<stdint.h>
#include<string>
#include<vector>

namespace utilities
{

	struct Image
	{
		uint32_t width;
		uint32_t heigth;
		std::vector<float> data;
		uint8_t nchannels;

	};

	bool loadImageRGB(std::string& path, Image& img);

	bool loadImageGrayScale(std::string& path, Image& img);

	bool saveImage(std::string& path, Image& img);
}