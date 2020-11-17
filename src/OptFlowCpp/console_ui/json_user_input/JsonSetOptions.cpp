#pragma once
#include"JsonSetOptions.h"

namespace console_ui
{
	std::shared_ptr<optflow_composition::PyramidSettings> ParsePyramid(modern_json::json input, std::shared_ptr<optflow_composition::PyramidSettings> options)
	{
		return options;
	}

	std::shared_ptr<optflow_solvers::GNCPenaltySolverSettings> ParseGNCPenaltySolver(modern_json::json input, std::shared_ptr<optflow_solvers::GNCPenaltySolverSettings> options)
	{
		return options;
	}

	std::shared_ptr<optflow_composition::PyramidIteratorSettings> ParsePyramidContainerSettings (modern_json::json input, std::shared_ptr<optflow_composition::PyramidIteratorSettings> options)
	{
		ParsePyramid(input, options->ConvexSettings);
		ParsePyramid(input, options->NonConvexSettings);
		return options;
	}

	std::shared_ptr<optflow_composition::CPUBackendOptions> ParseCpu(modern_json::json input, std::shared_ptr<optflow_composition::CPUBackendOptions> options)
	{
		return options;
	}

	std::shared_ptr<optflow_composition::FlowSolverOptions> ParseFlowSolver(modern_json::json input, std::shared_ptr<optflow_composition::FlowSolverOptions> options)
	{
		ParseGNCPenaltySolver(input, options->GNCSettings.GNCSettings);
		ParsePyramidContainerSettings(input, options->GNCSettings.PyramidContainerSettings);

		return options;
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
	}
}