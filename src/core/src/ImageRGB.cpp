#include "ImageRGB.h"
#include<OpenImageIO/imageio.h>
#include"logger.hpp"
#include"loggerHelper.hpp"
#include<filesystem>
#include<memory>

core::ImageRGB::ImageRGB(std::string filepath)
{

	auto input = OIIO::ImageInput::open(filepath);
	const OIIO::ImageSpec &spec = input->spec();
	width = spec.width;
	heigth = spec.height;
	
	nchannels = spec.nchannels;
	if (nchannels != 3)
	{
		width = -1;
		heigth = -1;
		return;
	}
	pixels = std::vector<unsigned char>(width*heigth*nchannels);
	input->read_image(OIIO::TypeDesc::UINT8, &pixels[0]);
	input->close();

}

core::ImageRGB::ImageRGB(uint32_t width, uint32_t height)
{
	this->width = width;
	this->heigth = height;

	nchannels = 3;
	pixels = std::vector<unsigned char>(width*heigth*nchannels,0);
}

core::Color core::ImageRGB::GetPixel(uint32_t x, uint32_t y) const
{
	size_t pixelIdx = x * nchannels + (heigth-y-1) * nchannels * width;
	core::Color color;
	color.Red = pixels[pixelIdx];

	color.Green = pixels[pixelIdx + 1];

	color.Blue = pixels[pixelIdx + 2];
	return color;
}

void core::ImageRGB::SetPixel(uint32_t x, uint32_t y, Color col)
{
	size_t pixelIdx = x * nchannels + (heigth - y - 1) * nchannels * width;
	pixels[pixelIdx] = col.Red;
	pixels[pixelIdx + 1] = col.Green;
	pixels[pixelIdx + 2] = col.Blue;
}

uint32_t core::ImageRGB::GetWidth() const
{
	return width;
}

uint32_t core::ImageRGB::GetHeight() const
{
	return heigth;
}

const unsigned char * core::ImageRGB::Data()
{
	return &pixels[0];
}

core::ImageRGB core::ImageRGB::Downsize(uint32_t target_wight, uint32_t target_height)
{
	return *this;
}

bool core::ImageRGB::save(std::string filepath)
{
	std::unique_ptr < OIIO::ImageOutput > out = OIIO::ImageOutput::create(filepath);
	if (!out)
	{
		return FALSE;
	}

	OIIO::ImageSpec  spec(width, heigth, nchannels, OIIO::TypeDesc::UINT8);
	
	out->open(filepath, spec);
	out->write_image(OIIO::TypeDesc::UINT8, Data());
	out->close();

	return TRUE;
}