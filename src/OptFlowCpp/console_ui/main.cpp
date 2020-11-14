#include"ComputeOpticalFlow.h"
#include "commandline_boost/ProgramOptions.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
	namespace bpo = boost::program_options;

	std::string first_img_path = "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png"; 
	//first_img_path = "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png";
	std::string second_img_path = "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png"; 
	//second_img_path = "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	std::string flow_output_path = "computed_flow.flo";
	std::string flow_img_path = "computed_img.png";
	
	auto di_container = console_ui::ComnandlineSetup(argc, argv, first_img_path, second_img_path, flow_output_path, flow_img_path);

	if (di_container == nullptr)
	{
		//input = --help

		//console_ui::ComputeOpticalFlow(first_img_path, second_img_path, flow_output_path, flow_img_path);

		return 1;
	}
	else
	{
		console_ui::ComputeOpticalFlow(first_img_path, second_img_path, flow_output_path, flow_img_path, di_container);
	}

	return 0;
}