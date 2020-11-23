#pragma once

#include <nlohmann/json.hpp>
#include <memory>
#include <string>

#include "optflow_composition/ContainerOptions.h"
#include "../../JSONHelper.h"

namespace json_settings
{
	namespace modern_json = nlohmann;

	void ParseIncrementalSolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::IncrementalSolverSettings> incremental_solver, const std::string string_end = "");

	void GenerateIncrementalSolver(modern_json::json& input, std::shared_ptr<optflow_solvers::IncrementalSolverSettings> incremental_solver, const std::string string_end = "");
}