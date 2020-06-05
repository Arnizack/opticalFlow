#pragma once
#include"IMinimizer.hpp"
#include"KDTree.h"
namespace cpu
{
	class BruteForceMinimizer: public IMinimizer 
	{
	public:
		BruteForceMinimizer(float sigma_d, float sigma_c, uint8_t sampleCount, uint8_t searchRegionSize);
		// Inherited via IMinimizer
		// Inherited via IMinimizer
		virtual core::FlowVector minimizeAtPixel(kdtree::KDTreeData& treeData, 
			const core::ImageRGB& templateFrame, const core::ImageRGB& nextFrame, 
			uint32_t x, uint32_t y, core::FlowVector& initialFlowVector) override;

	private:
		uint8_t SearchRegionSize;

		

		
	};

}
                