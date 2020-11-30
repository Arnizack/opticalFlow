
#include"opticalflow/opticalflow.h"
//#include <iostream>
#include <string>
//#include <vector>
#include"commandline_boost/ProgramOptions.h"

int main(int argc, char* argv[])
{
	/*
	* Peer 1 Image: "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png";
	* Christian 1 Image: "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame10.png";
	* Peer 2 Image: "E:\\dev\\opticalFlow\\V2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	* Christian 2 Image: "H:\\dev\\opticalFlow\\Prototyp\\Version 2\\opticalFlow\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	*/


	std::string first_img_path =  "..\\..\\..\\..\\resources\\eval-twoframes\\Dimetrodon\\frame10.png"; 
	std::string second_img_path = "..\\..\\..\\..\\resources\\eval-twoframes\\Dimetrodon\\frame11.png";
	
	std::string flow_output_path = "..\\..\\computed_flow.flo";
	std::string flow_img_path = "..\\..\\computed_img.png";

	std::string options_path = "";

	bool check_solve = console_ui::CheckCommandLineInput(argc, argv, first_img_path, second_img_path, flow_output_path, flow_img_path, options_path);

	if (check_solve == false)
		return 1;

	auto first_img = opticalflow::OpenImage(first_img_path);
	auto second_img = opticalflow::OpenImage(second_img_path);

	options_path = "..\\..\\console_ui\\temp_json_input.json";

	std::shared_ptr<opticalflow::OpticalFlowSolver> solver;

	if(options_path!="")
	{
		auto options = opticalflow::ReadOptions(options_path);

		solver = opticalflow::CreateSolver(options);
	}
	else
	{
		solver = opticalflow::CreateSolver();
	}
	auto result = solver->Solve(first_img,second_img);

	opticalflow::SaveFlowField(flow_output_path,result);
	opticalflow::SaveFlowFieldToColor(flow_img_path,result);

}