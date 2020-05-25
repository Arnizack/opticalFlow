#pragma once
#include"CoordinateConverter.hpp"
#include<stdint.h>
#include<array>
#include<vector>
#include"MortonNode.hpp"
#include"KdNode.h"
#include"KDTreeSpaces.h"
#include <stdexcept>

namespace kdtree
{

	struct KDTreeSample : public KdTreeVal
	{
		MortonNode* Parent = nullptr;

		KDTreeSample(KdTreeVal kdVal);

		KDTreeSample();

	};
	class KDTreeFactory
	{
	public:
		void sortMortonCodeAndSamples(std::vector<MortonCodeVal>& mortonCodes, std::vector<KDTreeSample>& samples);
	
		void makeMortonCodesUnique(std::vector<MortonCodeVal>& sortedCodes);
	
		int8_t findCommanPrefix(std::vector<MortonCodeVal>& mortonCodes, uint32_t a, uint32_t b);
	
		//Assumption: index < mortonCodes.size()-1
		short determineDirection(std::vector<MortonCodeVal>& mortonCodes, uint32_t index);
	
		std::array<uint32_t, 2> determineRange(std::vector<MortonCodeVal>& mortonCodes, uint32_t i);
	
		uint32_t findSplit(std::vector<MortonCodeVal>& mortonCodes, uint32_t first, uint32_t last);

		void buildMortonNodes(std::vector<MortonCodeVal>& mortonCodes, std::vector<MortonNode>& mortonNodes, std::vector<KDTreeSample>& samples);

		void computeBoundingBoxes(std::vector<MortonNode>& mortonNodes, std::vector<KDTreeSample>& samples);

		float findCut(CoordinateConverter& converter, uint64_t mortonCode, uint8_t prefix, short dimension);

		void createKdTreeNodes(std::vector<MortonNode>& mortonNodes, std::vector<MortonCodeVal>& mortonCodes, std::vector<KDTreeSample>& samples, std::vector<KdNode>& kdTreeNodes,
			CoordinateConverter& converter);

	};
}