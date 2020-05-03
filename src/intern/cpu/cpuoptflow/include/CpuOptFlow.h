#include<stdint.h>
#include"FlowField.h"
#include"ImageRGB.h"
#include<memory>

namespace cpu
{
	class OpticalFlow
	{
	public:
		OpticalFlow(float sigma_distance, float sigma_color, float convergence_threshold);
		
		core::FlowField CalcOptFlow(std::shared_ptr<core::ImageRGB> templateImage, std::shared_ptr<core::ImageRGB> nextImage);

	};
}