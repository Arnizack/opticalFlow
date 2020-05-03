#include "BilateralFlowFilter.h"
namespace  cpu::bilateralfilter
{
	core::FlowField BilateralFlowFilter::filter(const core::FlowField & flow, const ReliabilityMap & map, const kdtree::KDTree & tree)
	{
		return core::FlowField(0,0);
	}
}