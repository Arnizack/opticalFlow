#pragma once
#include"FlowContainerParser.h"

namespace json_settings
{
	void ParseFlowSolver(const modern_json::json& input, std::shared_ptr<optflow_composition::FlowSolverOptions> flow_solver, std::string string_end)
	{
		ParseGNCPenaltySolver(input, flow_solver->GNCSettings.GNCSettings, string_end);
		ParsePyramidContainerSettings(input, flow_solver->GNCSettings.PyramidContainerSettings, string_end);
	}

	void GenerateFlowSolver(modern_json::json& output, std::shared_ptr<optflow_composition::FlowSolverOptions> flow_solver, std::string string_end)
	{
		GenerateGNCPenaltySolver(output, flow_solver->GNCSettings.GNCSettings, string_end);
		GeneratePyramidContainerSettings(output, flow_solver->GNCSettings.PyramidContainerSettings, string_end);
	}
}