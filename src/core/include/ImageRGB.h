#pragma once
#include"Color.h"
#include <stdint.h>
#include<cuchar>

namespace core
{
	class ImageRGB
	{
	public:
		Color GetPixel(uint32_t x, uint32_t y) const;
		void SetPixel(uint32_t x, uint32_t y, Color col);
		uint32_t GetWidth() const;
		uint32_t GetHeight() const;

		const unsigned char* Data();

		ImageRGB Downsize(uint32_t target_wight, uint32_t target_height);
	};
}
