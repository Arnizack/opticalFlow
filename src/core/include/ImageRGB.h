#pragma once
#include"Color.h"
#include <stdint.h>
#include<cuchar>
#include<string>
#include<vector>

namespace core
{
	class ImageRGB
	{
	public:
		//can't import jpg correctly
		ImageRGB(std::string filepath);
		ImageRGB(uint32_t width, uint32_t height);
		virtual Color GetPixel(uint32_t x, uint32_t y) const;
		void SetPixel(uint32_t x, uint32_t y, Color col);
		uint32_t GetWidth() const;
		uint32_t GetHeight() const;

		const unsigned char* Data();

		ImageRGB Downsize(uint32_t target_wight, uint32_t target_height);
		
		//returns True, when saved successfull 
		bool Save(std::string filepath);

		/*
		normalize ensures, that the area under the kernel function is 1. 
		*/
		ImageRGB Conv1D (std::vector<float> kernel, bool yDirection = false, bool normalize = true);

		

	private:
		uint32_t width;
		uint32_t heigth;
		unsigned short nchannels;
		std::vector<unsigned char> pixels;

	};
}
