#pragma once
#include <boost/program_options.hpp>

#include <string>
#include <iostream>

namespace console_ui
{
	boost::program_options::options_description SetupGenericOptions();

	boost::program_options::options_description SetupIOOptions(std::string& first_image_path,
		std::string& second_image_path, std::string& flow_output_path, std::string& flow_img_output_path, std::string& json_input_path);

	boost::program_options::options_description SetupCGSolverOptions();

	boost::program_options::options_description SetupCPUBackendOptions();

	boost::program_options::options_description SetupCPULinalgOptions();

	boost::program_options::options_description SetupSunBakerSolverOptions();
}