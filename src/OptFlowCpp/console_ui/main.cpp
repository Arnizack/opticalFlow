#include"optflow_solvers/OpticalFlowApplication.h"
#include "commandline_boost/ProgramOptions.h"
#include"optflow_composition/ContainerInstaller.h"
#include "json_user_input/JsonSetOptions.h"

#include <iostream>
#include <string>
#include <vector>
#include"core/Logger.h"
#include"utilities/debug/ImageLogger.h"

int main(int argc, char* argv[])
{
	/*
	* Peer 1 Image: "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png";
	* Christian 1 Image: "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png";
	* Peer 2 Image: "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	* Christian 2 Image: "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	*/

	core::Logger::Init();
	debug::ImageLogger::Init(
		"E:\\dev\\opticalFlow\\optFlowCpp\\opticalFlow\\src\\OptFlowCpp\\bin\\debug_images"
		/*"H:\\dev\\opticalFlow\\optFlowCpp\\opticalFlow\\src\\OptFlowCpp\\bin\\debug_images"*/,
		"E:\\dev\\opticalFlow\\optFlowCpp\\opticalFlow\\src\\OptFlowCpp\\bin\\debug_flow"
		/*"H:\\dev\\opticalFlow\\optFlowCpp\\opticalFlow\\src\\OptFlowCpp\\bin\\debug_flow"*/);

	OF_LOG_INFO("Start Logger");

	std::string first_img_path = "..\\..\\..\\resources\\eval-twoframes\\Dimetrodon\\frame10.png"; 
	std::string second_img_path = "..\\..\\..\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	
	std::string flow_output_path = "computed_flow.flo";
	std::string flow_img_path = "computed_img.png";
	

	std::string json_input_path = "temp_json_input.json";
	
	//CommandLine Input
	/*bool check_solve = console_ui::CheckCommandLineInput(argc, argv, first_img_path, second_img_path, flow_output_path, flow_img_path, json_input_path);

	if (check_solve == false)
		return 1;*/

	optflow_composition::ContainerInstaller di_installer;

	auto options = std::make_shared<optflow_composition::ContainerOptions>();

	//Parse JSON
	console_ui::JsonSetupSettings(json_input_path, options);

	options->SolverOptions->GNCSettings.GNCSettings->GNCSteps = 1;

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