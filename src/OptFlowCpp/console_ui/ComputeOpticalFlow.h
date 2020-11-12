#pragma once
#include<string>

namespace console_ui
{

	

	void ComputeOpticalFlow(std::string first_image_path, std::string second_image_path, std::string flow_output_path, std::string flow_img_output_path);
}