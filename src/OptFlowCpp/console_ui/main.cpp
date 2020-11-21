#include"optflow_solvers/OpticalFlowApplication.h"
#include "commandline_boost/ProgramOptions.h"
#include"optflow_composition/ContainerInstaller.h"
#include "json_user_input/JsonSetOptions.h"

#include <iostream>
#include <string>
#include <vector>
#include"core/Logger.h"


int main(int argc, char* argv[])
{
	
	core::Logger::Init();

	OF_LOG_INFO("Start Logger");

	std::string first_img_path = "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png"; 
	//first_img_path = "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png";
	std::string second_img_path = "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png"; 
	//second_img_path = "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	std::string flow_output_path = "computed_flow.flo";
	std::string flow_img_path = "computed_img.png";

	std::string json_input_path = "temp_json_input.json"; //"E:\\dev\\opticalFlow\\OptFlowCpp\\opticalFlow\\src\\OptFlowCpp\\bin\\x64\\Release\\temp_json_input.json";
	
	//CommandLine Input
	/*bool check_solve = console_ui::CheckCommandLineInput(argc, argv, first_img_path, second_img_path, flow_output_path, flow_img_path, json_input_path);

	if (check_solve == false)
		return 1;*/

	optflow_composition::ContainerInstaller di_installer;

	auto options = std::make_shared<optflow_composition::ContainerOptions>();

	//Parse JSON
	console_ui::JsonSetupSettings(json_input_path, options);

	di_installer.SetOptions(options);

	auto di_container = di_installer.Install();

	auto application = di_container->resolve<optflow_solvers::OpticalFlowApplication>();

	application->ComputeOpticalFlow(first_img_path, second_img_path, flow_output_path, flow_img_path);

	/*
	* TEMP JSON Output
	*/

	/*nlohmann::json out_json = console_ui::GenerateJSON();

	console_ui::OutputJSON(out_json, "temp_json_input.json");*/

	return 0;
}