#include <boost/program_options.hpp>

#include"ComputeOpticalFlow.h"

#include <iostream>
#include <string>
#include <vector>

int set_mandatory_variables()
{
	std::cout << "See --help for must set variables\n";
	return 2;
}

int main(int argc, char* argv[])
{
	namespace bpo = boost::program_options;

	std::string first_img_path = "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png"; 
	//first_img_path = "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png";
	std::string second_img_path = "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png"; 
	//second_img_path = "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	std::string flow_output_path = "computed_flow.flo";
	std::string flow_img_path = "computed_img.png";

	
	////Setting up all posible Arguments
	//bpo::options_description generic("Generic options");
	//generic.add_options()
	//	("help", "produce help message");

	//bpo::options_description mandatory("Mandatory");
	//mandatory.add_options()
	//	("input-img1,1", bpo::value< std::string>(&first_img_path), "path to first input image")
	//	("input-img2,2", bpo::value< std::string>(&second_img_path), "path to second input image")
	//	("flow-img,F", bpo::value< std::string>(&flow_img_path), "path to flow image")
	//	("output-path,O", bpo::value< std::string>(&flow_output_path), "Output path");

	//bpo::options_description cmdline("All Options");
	//cmdline.add(generic).add(mandatory);

	////Parsing Comandline
	//bpo::variables_map vm;
	//bpo::store(bpo::parse_command_line(argc, argv, cmdline), vm);
	//bpo::notify(vm);

	//
	////Checking what is set
	//if (vm.count("help"))
	//{
	//	std::cout << cmdline << '\n';
	//	return 1;
	//}

	//if (!vm.count("input-img1"))
	//{
	//	set_mandatory_variables();
	//}

	//if (!vm.count("input-img2"))
	//{
	//	set_mandatory_variables();
	//}

	//if (!vm.count("flow-img"))
	//{
	//	set_mandatory_variables();
	//}

	//if (!vm.count("output-path"))
	//{
	//	set_mandatory_variables();
	//}

	////Test output
	//std::cout << "Input image 1 is: " << first_img_path << '\n';
	//std::cout << "Input image 2 is: " << second_img_path << '\n';
	//std::cout << "Output Path is: " << flow_output_path << '\n';
	//std::cout << "Flow image is: " << flow_img_path << '\n';
	
	console_ui::ComputeOpticalFlow(first_img_path, second_img_path, flow_output_path, flow_img_path);

	return 0;
}