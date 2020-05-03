#include "ImageRGB.h"

core::Color core::ImageRGB::GetPixel(uint32_t x, uint32_t y) const
{
	return core::Color();
}

void core::ImageRGB::SetPixel(uint32_t x, uint32_t y, Color col)
{
}

uint32_t core::ImageRGB::GetWidth() const
{
	return uint32_t();
}

uint32_t core::ImageRGB::GetHeight() const
{
	return uint32_t();
}

const unsigned char * core::ImageRGB::Data()
{
	return nullptr;
}

core::ImageRGB core::ImageRGB::Downsize(uint32_t target_wight, uint32_t target_height)
{
	return *this;
}
