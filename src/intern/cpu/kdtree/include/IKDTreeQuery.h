#pragma once
#include"KdNode.h"
#include"KDTreeSpaces.h"
#include<vector>
#include"KDResults.h"
namespace kdtree
{
	class Test
	{
	public:
		float test;
	};

	class IKDTreeQuery
	{
	public:
		
		virtual void Query(KdTreeVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results,
			float sigma, uint8_t sampleCount) = 0;
		virtual void Query(StandardVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results,
			float sigma, uint8_t sampleCount) = 0;

	};

}