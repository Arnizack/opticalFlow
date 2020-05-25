#pragma once
#include"IKDTreeQuery.h"
#include"KDTreeMath.hpp"
#include"IKDRandom.hpp"
#include"KDSerialRandom.hpp"

namespace kdtree
{
	class KDTreeRecrusivQuery : public IKDTreeQuery
	{
	public:
		KDTreeRecrusivQuery(uint32_t weight, uint32_t height, uint8_t maxSampleCount, CoordinateConverter converter);
		void Query(KdTreeVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results,
			float sigma, uint8_t sampleCount) override;
		
		void Query(StandardVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results,
			float sigma, uint8_t sampleCount) override;

	private:
		void QueryNode(KdTreeVal& point, uint32_t nodeIdx, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples,
			std::vector<KDResult>& results, float sigma, 
			uint8_t sampleStart,uint8_t sampleEnd, float p=1, int level = 0);
		void QueryLeaf(KdTreeVal& point, uint32_t leafIdx, const std::vector<KdNode>& kdNodes, 
			const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma, uint8_t sampleStart, uint8_t sampleEnd, float p = 1, int level = 0);

		CoordinateConverter Converter;
		std::unique_ptr<IKDRandom> Generator;

	};
}