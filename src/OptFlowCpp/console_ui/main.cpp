#include <boost/program_options.hpp>

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
	namespace bpo = boost::program_options;

	std::string input_file_1;
	std::string input_file_2;
	std::string output_path;

	//Setting up all posible Arguments
	bpo::options_description generic("Generic options");
	generic.add_options()
		("help", "produce help message");

	bpo::options_description mandatory("Mandatory");
	mandatory.add_options()
		("input-file1,1", bpo::value< std::string>(&input_file_1), "first input file")
		("input-file2,2", bpo::value< std::string>(&input_file_2), "second input file")
		("output-path,O", bpo::value< std::string>(&output_path), "Output path");

	bpo::options_description cmdline("All Options");
	cmdline.add(generic).add(mandatory);

	//Parsing Comandline
	bpo::variables_map vm;
	bpo::store(bpo::parse_command_line(argc, argv, cmdline), vm);
	bpo::notify(vm);

	//Checking what is set
	if (vm.count("help"))
	{
		std::cout << cmdline << '\n';
		return 1;
	}

	if (!vm.count("input-file1"))
	{
		std::cout << "See --help for must set variables\n";
		return 2;
	}

	if (!vm.count("input-file2"))
	{
		std::cout << "See --help for must set variables\n";
		return 2;
	}

	if (!vm.count("output-path"))
	{
		std::cout << "See --help for must set variables\n";
		return 2;
	}

	//Test output
	std::cout << "Include File 1 is: " << input_file_1 << '\n';
	std::cout << "Include File 2 is: " << input_file_2 << '\n';
	std::cout << "Output Path is: " << output_path << '\n';

	return 0;
#include"ComputeOpticalFlow.h"

int main()
{
	std::string first_img_path = "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png";
	std::string second_img_path = "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	std::string flow_output_path = "computed_flow.flo";
	std::string flow_img_path = "computed_img.png";

	console_ui::ComputeOpticalFlow(first_img_path, second_img_path, flow_output_path, flow_img_path);
}