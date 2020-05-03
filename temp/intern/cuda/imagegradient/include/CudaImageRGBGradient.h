#pragma once

#include<stdint.h>
#include"ICUDATransferable.h"
#include"ImageRGBGradient.h"

struct CudaGradientData;

__device__ float[2][3] getGradient(uint32_t x, uint32_t y, CudaGradientData data);

__device__ void setGradient(float[2][3] gradient, unint32_t x, uint32_t y, CudaGradientData data);

class CudaImageRGBGradient : ICUDATransferable<CudaGradientData>
{
public:
	CudaImageRGBGradient(ImageRGBGradient gradient);

	CudaImageRGBGradient(uint32_t weigth, uint32_t height);

	CudaGradientData GetDataForGPU();

};

