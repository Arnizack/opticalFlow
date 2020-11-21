#pragma once
#include"JsonSetOptions.h"

namespace console_ui
{
	modern_json::json GenerateJSON()
	{
		std::shared_ptr<optflow_composition::ContainerOptions> options = std::make_shared<optflow_composition::ContainerOptions>(optflow_composition::ContainerOptions());

		modern_json::json output;

		GenerateCpu(output, options->CPUOptions);
		GenerateFlowSolver(output, options->SolverOptions);

		return output;
	}

	std::shared_ptr<optflow_composition::ContainerOptions> JsonSetupSettings(std::string json_input_file_path, std::shared_ptr<optflow_composition::ContainerOptions> options)
	{
		//read in
		std::ifstream file(json_input_file_path);
		modern_json::json input;

		if (!file.is_open())
			return options;

		file >> input;

		ParseCpu(input, options->CPUOptions);
		ParseFlowSolver(input, options->SolverOptions);

		return options;
	}
}