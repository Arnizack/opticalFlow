#pragma once
#include"FlowField.h"
#include"ImageRGB.h"
#include<KDTree.h>
namespace cpu
{
	class IMinimizer
	{
	public:
		IMinimizer(float sigma_d, float sigma_c, uint8_t sampleCount);
		core::FlowField minimize(std::unique_ptr<core::FlowField>& initialFlow, const std::shared_ptr<core::ImageRGB>& templateFrame,
			const std::shared_ptr<core::ImageRGB>& nextFrame);
	
	protected:
		float Sigma_c;
		float Sigma_d;
		uint8_t SampleCount;
	private:
		virtual core::FlowVector minimizeAtPixel(kdtree::KDTreeData& treeData,const core::ImageRGB& templateFrame, 
			const core::ImageRGB& nextFrame, uint32_t x, uint32_t y, core::FlowVector& initialFlowVector) = 0;
		
	};

}
                