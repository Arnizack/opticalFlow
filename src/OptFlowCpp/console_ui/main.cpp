#include"optflow_solvers/OpticalFlowApplication.h"
//#include "commandline_boost/ProgramOptions.h"
#include"optflow_composition/ContainerInstaller.h"
#include <iostream>
#include <string>
#include <vector>
#include"core/Logger.h"


int main(int argc, char* argv[])
{
	
	core::Logger::Init();

	OF_LOG_INFO("Start Logger");

	std::string first_img_path = "..\\..\\..\\resources\\eval-twoframes\\Dimetrodon\\frame10.png"; 
	std::string second_img_path = "..\\..\\..\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	std::string flow_output_path = "computed_flow.flo";
	std::string flow_img_path = "computed_img.png";
	
	optflow_composition::ContainerInstaller di_installer;

	auto options = std::make_shared<optflow_composition::ContainerOptions>();
	di_installer.SetOptions(options);

	auto di_container = di_installer.Install();

	auto application = di_container->resolve<optflow_solvers::OpticalFlowApplication>();

	application->ComputeOpticalFlow(first_img_path, second_img_path, flow_output_path, flow_img_path);

	return 0;
}