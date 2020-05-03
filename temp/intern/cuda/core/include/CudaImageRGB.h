#pragma once
#include"ICUDATransferable.h"
#include"ImageRGB.h"

struct CUDAImageData;

class CudaImageRGB : public ICUDATransferable<CUDAImageData>
{
public:
	CUDAImageData GetDataForGPU();
	
	CudaImageRGB(const ImageRGB& image);
	CudaImageRGB(uint32_t wight, uint32_t height);

	~CudaImageRGB();
	
	ImageRGB GetImageToCPU();


	CudaImageRGB Downsize(uint32_t target_wight, uint32_t target_height);
};

