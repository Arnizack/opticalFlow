#pragma once

#include"CUDAFlowField.h"
#include"CUDAKDTree.h"
#include"CudaReliabilityMap.h"


class CudaBilateralFlowFilter
{
public:
	static CudaFlowField filter(const CudaFlowField& flow, const CudaReliabilityMap& map, const CUDAKDTree& tree);
};



