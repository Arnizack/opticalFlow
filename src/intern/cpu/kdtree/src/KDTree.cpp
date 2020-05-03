#include"KDTree.h"

namespace kdtree
{
	struct KDTreeData
	{

	};

	std::vector<KDResult> queryKDTree(KDTreeData data)
	{
		return std::vector<KDResult>();
	}

	KDTree::KDTree(std::shared_ptr<core::ImageRGB> image)
	{
	}

	KDTreeData KDTree::Build(float sigma_distance, float sigma_color, int sampleCount)
	{
		return KDTreeData();
	}
}