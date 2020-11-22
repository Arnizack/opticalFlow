#pragma once
#include"JSONHandler.h"


namespace json_settings
{
	void JsonSetupSettings(std::string json_input_file_path, std::shared_ptr<optflow_composition::ContainerOptions> options)
	{
		//read in
		std::ifstream file(json_input_file_path);
		modern_json::json input;

		if (!file.is_open())
			return;

		file >> input;

		ParseCpu(input["cpu_settings"], options->CPUOptions);
		ParseFlowSolver(input["solver_settings"], options->SolverOptions);

		return;
	}

	modern_json::json GenerateJSON()
	{
		std::shared_ptr<optflow_composition::ContainerOptions> options = std::make_shared<optflow_composition::ContainerOptions>(optflow_composition::ContainerOptions());

		modern_json::json output;

		output["cpu_settings"];
		GenerateCpu(output["cpu_settings"], options->CPUOptions);

		output["solver_settings"];
		GenerateFlowSolver(output["solver_settings"], options->SolverOptions);

		return output;
	}
}