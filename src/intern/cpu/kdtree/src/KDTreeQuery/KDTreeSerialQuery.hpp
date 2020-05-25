#pragma once
#include"IKDTreeQuery.h"
#include <CoordinateConverter.hpp>
#include"IKDRandom.hpp"

namespace kdtree
{
	class KDTreeSerialQuery : public IKDTreeQuery
	{

	public:
		KDTreeSerialQuery(uint32_t weight, uint32_t height, uint8_t maxSampleCount, CoordinateConverter converter);

		// Inherited via IKDTreeQuery
		void Query(KdTreeVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma, uint8_t sampleCount) override;
		void Query(StandardVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma, uint8_t sampleCount) override;
	private:

		KDResult querySingle(KdTreeVal& point, uint32_t sampleIdx, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples,
			float sigma,
			uint8_t sampleStart, uint8_t sampleEnd, float p = 1, int level = 0);

		CoordinateConverter Converter;
		std::unique_ptr<IKDRandom> Generator;
	
	};

}
                