#pragma once
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

#include "optflow_composition/FlowSolverOptions.h"
#include "PyramidContainerParser.h"
#include "FlowSolverParser/GNCPenaltySolverParser.h"

namespace json_settings
{
	namespace modern_json = nlohmann;

	void ParseFlowSolver(const modern_json::json& input, std::shared_ptr<optflow_composition::FlowSolverOptions> flow_solver, std::string string_end = "");

	void GenerateFlowSolver(modern_json::json& output, std::shared_ptr<optflow_composition::FlowSolverOptions> flow_solver, std::string string_end = "");
}