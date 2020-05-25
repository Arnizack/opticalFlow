#include "KDTreeSerialQuery.hpp"
#include"KDSerialRandom.hpp"
#include"KDTreeMath.hpp"

namespace kdtree
{
    KDTreeSerialQuery::KDTreeSerialQuery(uint32_t weight, uint32_t height, uint8_t maxSampleCount, CoordinateConverter converter)
        : Converter{converter}
	{
		Generator = std::make_unique<KDSerialRandom>(weight, height, maxSampleCount);
	}

    void KDTreeSerialQuery::Query(KdTreeVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma, uint8_t sampleCount)
    {
		
		for (uint8_t sampleIdx = 0; sampleIdx < sampleCount; sampleIdx++)
		{
			KDResult result = querySingle(point, sampleIdx, kdNodes, samples, sigma, 0, sampleCount);
			results.push_back(result);
		}
    }

	void KDTreeSerialQuery::Query(StandardVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma, uint8_t sampleCount)
	{
		KdTreeVal kdPoint = Converter.StandardToKdTree(point);
		Query(kdPoint, kdNodes, samples, results, sigma, sampleCount);
	}

    KDResult KDTreeSerialQuery::querySingle(KdTreeVal& point, uint32_t sampleIdx, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, 
		float sigma, uint8_t sampleStart, uint8_t sampleEnd, float p, int level)
    {
        uint32_t nodeIdx = 0;
		uint32_t leafIdx = -1;
		uint8_t sampleCount;
		StandardVal pointStandard = Converter.KdTreeToStandard(point);
		do
		{

			sampleCount = sampleEnd - sampleStart;
			const auto& node = kdNodes[nodeIdx];
			float dimValue = point[node.Dimension];

			float cdfMin = CdfApprox((node.Min - dimValue) / sigma);
			float cdfMax = CdfApprox((node.Max - dimValue) / sigma);

			float cdfCut = CdfApprox((node.Cut - dimValue) / sigma);

			float pLeft = (cdfCut - cdfMin) / (cdfMax - cdfMin);
			float expectedLeft = pLeft * sampleCount;
			uint8_t samplesLeft = Floor(expectedLeft);
			uint8_t samplesRight = Floor(sampleCount - expectedLeft);


			if (samplesLeft + samplesRight < sampleCount)
			{
				
				if (Generator->Urand(nodeIdx, pointStandard.X, pointStandard.Y) < expectedLeft - samplesLeft)
					samplesLeft++;
				else
					samplesRight++;
			}

			uint8_t sampleMiddle = sampleStart + samplesLeft;


			if (sampleMiddle > sampleIdx)
			{
				//Left
				sampleEnd = sampleMiddle;
				if (node.IsLeftLeaf)
					leafIdx = node.Left;
				else
					nodeIdx = node.Left;

			}
			else
			{
				//Rigth
				sampleStart = sampleMiddle;
				if (node.IsRigthLeaf)
					leafIdx = node.Rigth;
				else
					nodeIdx = node.Rigth;
			}
			level++;

		} while (leafIdx==static_cast<uint32_t>(-1));

		KdTreeVal resultKd = samples[leafIdx];
		float distance = DistanceSquared(point, resultKd);
		float weight = static_cast<float>(exp(-distance / (2 * sigma)));

		StandardVal resultStandard = Converter.KdTreeToStandard(resultKd);

		return KDResult(resultStandard.X, resultStandard.Y, resultStandard.R, resultStandard.G, resultStandard.B, weight, level);
    }
}
