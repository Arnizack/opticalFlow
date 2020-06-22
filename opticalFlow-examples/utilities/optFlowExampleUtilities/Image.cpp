#include"Image.h"
#include<OpenImageIO/imageio.h>


template<size_t nchannels>
bool loadImage(std::string& path, utilities::Image& img)
{
	auto input = OIIO::ImageInput::open(path);

	if (!input)
		return false;

	const OIIO::ImageSpec& spec = input->spec();
	if (!spec.nchannels != nchannels)
		return false;

	img.width = spec.width;
	img.heigth = spec.height;
	img.data = std::vector<float>(img.width * img.heigth * nchannels);
	input->read_image(OIIO::TypeDesc::FLOAT, img.data.data());
	input->close();
	img.nchannels = nchannels;
	return true;
}

bool utilities::loadImageRGB(std::string& path, utilities::Image& img)
{
	return loadImage<3>(path, img);
}

bool utilities::loadImageGrayScale(std::string& path, Image& img)
{
	return loadImage<1>(path, img);
}

bool utilities::saveImage(std::string& path, utilities::Image& img)
{
	if (img.nchannels != 3 || img.nchannels != 1)
		return false;
	std::unique_ptr < OIIO::ImageOutput > out = OIIO::ImageOutput::create(path);
	if (!out)
		return false;
	OIIO::ImageSpec spec(img.width, img.heigth, img.nchannels, OIIO::TypeDesc::FLOAT);
	out->open(path, spec);
	out->write_image(OIIO::TypeDesc::UINT8, img.data.data());
	out->close();
	return true;
}


