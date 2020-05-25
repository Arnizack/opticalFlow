#include "ImageRGB.h"
#include<OpenImageIO/imageio.h>
#include"logger.hpp"
#include"loggerHelper.hpp"
#include<filesystem>
#include<memory>
#include<cmath>
#include<array>

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

float gaussianPdf(float standardDeviation, float x)
{
	const float pi = M_PI;
	float premulticator = 1 /( standardDeviation * std::sqrt(2 * pi));
	float exponent = -x * x / (2 * standardDeviation*standardDeviation);
	return premulticator * std::exp(exponent);
}

std::vector<float> createGaussianKernel(float standardDeviation, size_t kernelSize)
{
	std::vector<float> kernel(kernelSize);
	
	for (int idx = 0; idx < kernelSize; idx++)
	{
		kernel[idx] = gaussianPdf(standardDeviation, idx - (int)(kernelSize / 2));
	}

	return kernel;
}


core::ImageRGB core::ImageRGB::Conv1D(std::vector<float> kernel,bool yDirection, bool normalize)
{
	
	core::ImageRGB bluredImg(width,heigth);

	int32_t kernelHalf = kernel.size() / 2;

	


	for (uint32_t x = 0; x < width; x++)
	{
		for (uint32_t y = 0; y < heigth; y++)
		{
			float kernelSum = 0;
			float pixelValueR = 0;
			float pixelValueG = 0;
			float pixelValueB = 0;
			for (int kernelIdx = 0; kernelIdx < kernel.size(); kernelIdx++)
			{

				if (yDirection)
				{
					if (y+kernelIdx >= kernelHalf)
					{
						uint32_t pixelInKernelY = y + kernelIdx - kernelHalf;

						if(pixelInKernelY<heigth)
						{ 

							uint32_t pixelInKernelX = x;

							core::Color pixelInKernel = this->GetPixel(pixelInKernelX, pixelInKernelY);

							float kernelValue = kernel[kernelIdx];

							pixelValueR += kernelValue * pixelInKernel.Red;
							pixelValueG += kernelValue * pixelInKernel.Green;
							pixelValueB += kernelValue * pixelInKernel.Blue;
							
							if(normalize)
								kernelSum += std::abs(kernelValue);
						}

					}
				}
				else
				{
					if (x + kernelIdx >= kernelHalf)
					{
						uint32_t pixelInKernelX = x + kernelIdx - kernelHalf;

						if(pixelInKernelX<width)
						{
						
							uint32_t pixelInKernelY = y;

							core::Color pixelInKernel = this->GetPixel(pixelInKernelX, pixelInKernelY);

							float kernelValue = kernel[kernelIdx];

							pixelValueR += kernelValue * pixelInKernel.Red;
							pixelValueG += kernelValue * pixelInKernel.Green;
							pixelValueB += kernelValue * pixelInKernel.Blue;
						}
					}
				}
			}
			if(normalize)
			{
				pixelValueR /= kernelSum;
				pixelValueB /= kernelSum;
				pixelValueG /= kernelSum;
			}
			core::Color resultColor;
			if (pixelValueR > 225)
				pixelValueR = 225;
			if (pixelValueB > 225)
				pixelValueB = 225;
			if (pixelValueG > 225)
				pixelValueG = 225;
			resultColor.Red = pixelValueR;
			resultColor.Green = pixelValueG;
			resultColor.Blue = pixelValueB;
			bluredImg.SetPixel(x, y, resultColor);
			
		}
	}

	return bluredImg;
}

core::ImageRGB core::ImageRGB::Downsize(uint32_t target_wight, uint32_t target_height)
{
	float scaleFactorX = (float)width/(float)target_wight  ;
	float scaleFactorY = (float)heigth/(float) target_height  ;

	float standardDeviationX = 1 / std::sqrt(2 * scaleFactorX);
	float standardDeviationY = 1 / std::sqrt(2 * scaleFactorY);

	int kernelSizeX = std::round(3 * standardDeviationX) * 2 + 1 ;
	int kernelSizeY = std::round(3 * standardDeviationY) * 2 + 1;

	
	auto kernelX = createGaussianKernel(standardDeviationX, kernelSizeX);
	auto kernelY = createGaussianKernel(standardDeviationY, kernelSizeY);

	this->Save("input.png");

	core::ImageRGB xBlurImg = Conv1D(kernelX);

	core::ImageRGB xyBlurImg = xBlurImg.Conv1D(kernelY,true);

	core::ImageRGB downScaledImg(target_wight, target_height);

	for (int x = 0; x < target_wight; x++)
	{
		for (int y = 0; y < target_height; y++)
		{
			int upScaledX = (int) x * scaleFactorX;
			int upScaledY = (int) y * scaleFactorY;

			core::Color color = xyBlurImg.GetPixel(upScaledX, upScaledY);
			downScaledImg.SetPixel(x, y, color);
		}
	}

	return downScaledImg;
}

bool core::ImageRGB::Save(std::string filepath)
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