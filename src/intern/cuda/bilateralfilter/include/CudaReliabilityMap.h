#pragma once
#include"CUDAImageRGB.h"
#include"CUDAFlowField.h"
#include"ICUDATransferable.h"


struct CudaReliabilityMapData;

__device__ float GetReliability(CudaReliabilityMapData data, uint32_t x, uint32_t y);

class CudaReliabilityMap : public ICUDATransferable<CudaReliabilityMapData>
{
public:
	CudaReliabilityMap(const FlowField& flow, const ImageRGB& templateFrame, const ImageRGB& nextFrame);
	~CudaReliabilityMap();

	CudaReliabilityMapData GetDataForGPU();

	
};

