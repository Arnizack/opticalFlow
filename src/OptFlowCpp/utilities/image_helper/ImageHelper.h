#pragma once
#include"ImageHelper.h"
#include"core/IArray.h"
#include<memory>
#include<vector>
#include<string>

namespace imagehelper
{

	struct Image
	{
		size_t width;
		size_t height;
		size_t color_count;
		std::shared_ptr<std::vector<float>> data;
	};

	void SaveImage(std::string filepath, float* img, size_t width, size_t height, size_t color_count);
	void SaveImage(std::string filepath, Image img);
	void SaveImage(std::string filepath, std::shared_ptr<core::IArray<float, 2>> img);
	void SaveImage(std::string filepath, std::shared_ptr < core::IArray<float, 3>> img);
	
	
	Image OpenImage(std::string filepath);

}