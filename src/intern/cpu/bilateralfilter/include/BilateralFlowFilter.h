#pragma once
#include"FlowField.h"
#include"KDTree.h"
#include"ReliabilityMap.h"

namespace cpu::bilateralfilter
{

	class BilateralFlowFilter
	{
	public:
		static core::FlowField filter(const core::FlowField& flow, const ReliabilityMap& map, const kdtree::KDTree& tree);
	};

}
