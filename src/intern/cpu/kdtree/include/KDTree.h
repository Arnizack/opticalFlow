#pragma once
#include<memory>
#include"ImageRGB.h"
#include<vector>

namespace kdtree
{ 

	struct KDTreeData;

	struct KDResult
	{
		float X;
		float Y;
		float R;
		float G;
		float B;
	};

	std::vector<KDResult> queryKDTree(KDTreeData data);

	class KDTree
	{
	public:
		KDTree(std::shared_ptr<core::ImageRGB> image);
		KDTreeData Build(float sigma_distance, float sigma_color,int sampleCount);
		
		
	};

}
