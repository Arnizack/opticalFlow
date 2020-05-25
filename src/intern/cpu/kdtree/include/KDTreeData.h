#pragma once
#include"KDTreeSpaces.h"
#include"IKDTreeQuery.h"
#include"KdNode.h"
#include<memory>
namespace kdtree
{

	struct KDTreeData
	{
	public:
		KDTreeData(std::unique_ptr<IKDTreeQuery> query);

		float Sigma_distance;
		float Sigma_color;
		int sampleCount;
		std::vector<KdTreeVal> samples;
		std::vector<KdNode> kdTreeNodes;
		std::unique_ptr<kdtree::IKDTreeQuery> Query;
	};
}