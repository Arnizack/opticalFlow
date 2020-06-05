#include"CpuOptFlow.h"

namespace cpu {
	OpticalFlow::OpticalFlow(float sigma_distance, float sigma_color, float convergence_threshold)
	{
	}

	core::FlowField OpticalFlow::CalcOptFlow(std::shared_ptr<core::ImageRGB> templateImage, std::shared_ptr<core::ImageRGB> nextImage)
	{
		return core::FlowField(0,0);
	}
}