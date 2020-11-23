#pragma once

#include <nlohmann/json.hpp>
#include <memory>
#include <string>

#include "optflow_composition/ContainerOptions.h"

#include "../../JSONHelper.h"

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName(var) (#var)

namespace console_ui
{
	namespace modern_json = nlohmann;

	void ParseCGSolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::CGSolverSettings> cg_solver, const std::string string_end = "");

	void GenerateCGSolver(modern_json::json& input, std::shared_ptr<optflow_solvers::CGSolverSettings> cg_solver, const std::string string_end = "");
}