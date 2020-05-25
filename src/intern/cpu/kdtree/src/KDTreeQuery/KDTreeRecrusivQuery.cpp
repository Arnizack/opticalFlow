#include"KDTreeRecrusivQuery.hpp"
#include <iostream>
#include"loggerHelper.hpp"
#include"Timer.h"

namespace kdtree
{

KDTreeRecrusivQuery::KDTreeRecrusivQuery(uint32_t weight, uint32_t height, uint8_t maxSampleCount, CoordinateConverter converter ) 
	: Converter(converter)
{
	Generator = std::make_unique<KDSerialRandom>(weight, height, maxSampleCount);
}

void KDTreeRecrusivQuery::Query(KdTreeVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma, uint8_t sampleCount)
{
	QueryNode(point, 0, kdNodes, samples, results, sigma, 0, sampleCount);
}

void KDTreeRecrusivQuery::Query(StandardVal point, const std::vector<KdNode>& kdNodes, const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma, uint8_t sampleCount)
{
	KdTreeVal kdPoint =  Converter.StandardToKdTree(point);
	Query(kdPoint, kdNodes, samples, results, sigma, sampleCount);
}

void KDTreeRecrusivQuery::QueryNode(KdTreeVal& point, uint32_t nodeIdx, const std::vector<KdNode>& kdNodes, 
	const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma,
	uint8_t sampleStart, uint8_t sampleEnd, float p, int level)
{
	
	uint8_t sampleCount = sampleEnd - sampleStart;
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
		StandardVal pointStandard = Converter.KdTreeToStandard(point);
		if (Generator->Urand(nodeIdx, pointStandard.X, pointStandard.Y) < expectedLeft - samplesLeft)
			samplesLeft++;
		else
			samplesRight++;
	}
	
	uint8_t sampleMiddle = sampleStart + samplesLeft;

	
	if (samplesLeft > 0)
	{
		uint32_t leftChildIdx = node.Left;
		//left Query
		if (node.IsLeftLeaf == false)
		{
			
			
			QueryNode(point, leftChildIdx, kdNodes, samples, results, sigma, sampleStart, sampleMiddle, p * pLeft, level + 1);
		}
		else
		{
		
			QueryLeaf(point, leftChildIdx, kdNodes, samples, results, sigma, sampleStart, sampleMiddle,p * pLeft,level+1);
		}
	}
	if (samplesRight > 0)
	{
		uint32_t rightChildIdx = node.Rigth;
		if (node.IsRigthLeaf == false)
		{
			QueryNode(point, rightChildIdx, kdNodes, samples, results, sigma, sampleMiddle, sampleEnd, p * (1 - pLeft),level+1);
		}
		else
		{
			QueryLeaf(point, rightChildIdx, kdNodes, samples, results, sigma, sampleMiddle, sampleEnd, p * (1 - pLeft),level+1);
		}
	}
	
}

void KDTreeRecrusivQuery::QueryLeaf(KdTreeVal& point, uint32_t leafIdx, const std::vector<KdNode>& kdNodes,
	const std::vector<KdTreeVal>& samples, std::vector<KDResult>& results, float sigma, uint8_t sampleStart, uint8_t sampleEnd, float p, int level)
{
	
	int sampleCount = sampleEnd - sampleStart;
	
	KdTreeVal resultKd = samples[leafIdx];
	
	float distance = DistanceSquared(point, resultKd);

	float weight = static_cast<float>( exp(-distance / (2 * sigma)));
	weight *= sampleCount;
	
	StandardVal resultStandard = Converter.KdTreeToStandard(resultKd);	
	
	results.emplace_back(resultStandard.X, resultStandard.Y, resultStandard.R, resultStandard.G, resultStandard.B, weight,level);
}

}

