#pragma once
#include "SetupOptions.h"
#include <boost/program_options.hpp>
#include <string>
#include <iostream>
#include"Hypodermic/ContainerBuilder.h"

namespace console_ui
{
	namespace bpo = boost::program_options;

	bool CheckCommandLineInput(int argc, char* argv[],
		std::string& first_image_path, std::string& second_image_path,
		std::string& flow_output_path, std::string& flow_img_output_path, 
		std::string& json_input_path);

	bpo::variables_map ParseCommandline(int argc, char* argv[], bpo::options_description options);
}