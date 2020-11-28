
#include"opticalflow/opticalflow.h"
//#include <iostream>
#include <string>
//#include <vector>

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

	std::string options_path = "..\\..\\console_ui\\temp_json_input.json";

	auto first_img = opticalflow::OpenImage(first_img_path);
	auto second_img = opticalflow::OpenImage(second_img_path);
	
	auto options = opticalflow::ReadOptions(options_path);

	auto solver = opticalflow::CreateSolver(options);

	auto result = solver->Solve(first_img,second_img);

	opticalflow::SaveFlowField(flow_output_path,result);
	opticalflow::SaveFlowFieldToColor(flow_img_path,result);

}