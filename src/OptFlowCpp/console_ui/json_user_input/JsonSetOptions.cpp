#pragma once
#include"JsonSetOptions.h"

namespace console_ui
{
	////Solver und Cpu
	//void ParseCGSolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::CGSolverSettings> options)
	//{
	//}

	//void ParseIncrementalSolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::IncrementalSolverSettings> options)
	//{
	//}

	//void ParseLinearizationSolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::LinearizationSolverSettings> options)
	//{
	//}

	//void ParseGNCPenaltySolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::GNCPenaltySolverSettings> options)
	//{
	//}

	//void ParseCpu(const modern_json::json& input, std::shared_ptr<optflow_composition::CPUBackendOptions> options)
	//{
	//}


	////remaining tree
	//void ParseLevelSettings(const modern_json::json& input, optflow_composition::LevelSettings& options)
	//{
	//	ParseLinearizationSolver(input, options.Linearization);
	//	ParseIncrementalSolver(input, options.Incremental);
	//	ParseCGSolver(input, options.CGsolver);
	//}

	//void ParsePyramidResolution(const modern_json::json& input, optflow_composition::PyramidResolution& options)
	//{
	//}

	//void ParsePyramid(const modern_json::json& input, optflow_composition::PyramidSettings& options)
	//{
	//	ParseLevelSettings(input, options.SingleLevelSettings);
	//	ParsePyramidResolution(input, options.Resolution);
	//}

	//void ParsePyramidContainerSettings (const  modern_json::json& input, optflow_composition::PyramidIteratorSettings& options)
	//{
	//	ParsePyramid(input, options.ConvexSettings);
	//	ParsePyramid(input, options.NonConvexSettings);
	//}

	//void ParseFlowSolver(const modern_json::json& input, std::shared_ptr<optflow_composition::FlowSolverOptions> options)
	//{
	//	ParseGNCPenaltySolver(input, options->GNCSettings.GNCSettings);
	//	ParsePyramidContainerSettings(input, options->GNCSettings.PyramidContainerSettings);
	//}


	modern_json::json GenerateJSON()
	{
		std::shared_ptr<optflow_composition::ContainerOptions> options = std::make_shared<optflow_composition::ContainerOptions>(optflow_composition::ContainerOptions());

		modern_json::json output;

		GenerateCpu(output, options->CPUOptions);
		GenerateFlowSolver(output, options->SolverOptions);

		return output;
	}

	void OutputJSON(const modern_json::json& output, const std::string& file_path)
	{
		std::ofstream out_stream(file_path);

		out_stream << std::setw(output.size() * 2) << output << std::endl;
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