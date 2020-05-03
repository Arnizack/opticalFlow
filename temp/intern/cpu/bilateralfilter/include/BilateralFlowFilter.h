#pragma once
#include"FlowField.h"
#include"KDTree.h"
#include"ReliabilityMap.h"
class BilateralFlowFilter
{
public:
	static FlowField filter(const FlowField& flow, const ReliabilityMap& map, const KDTree& tree);
};

