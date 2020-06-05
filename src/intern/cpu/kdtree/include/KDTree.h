#pragma once
#include<memory>
#include"ImageRGB.h"
#include<vector>
#include"KDTreeData.h"
#include"KDResults.h"
namespace kdtree
{ 

	std::vector<KDResult> queryKDTree(kdtree::KDTreeData& data,uint32_t x, uint32_t y, core::Color color);

	class KDTree
	{
	public:
		KDTree(std::shared_ptr<core::ImageRGB> image);
		kdtree::KDTreeData Build(float sigma_distance, float sigma_color, int sampleCount);
		
	private:
		std::shared_ptr<core::ImageRGB> Image;
	};

}
