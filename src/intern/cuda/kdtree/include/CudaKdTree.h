#pragma once
#include"KDTree.h"
#include"CUDAImageRGB.h"
#include<memory>

struct CUDAKDTreeData;

namespace CUDAKDTree
{

KDResult queryKDTree(CUDAKDTreeData data);
}
class CUDAKDTree
{
public:
	CUDAKDTree(std::shared_ptr<CudaImageRGB> image);

	CUDAKDTreeData Build(float sigma_distance, float sigma_color, int sampleCount);
};

