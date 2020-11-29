#pragma once

#include <nlohmann/json.hpp>
#include <memory>
#include <string>

#include "optflow_composition/ContainerOptions.h"
#include "../../JSONHelper.h"

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName


namespace json_settings
{
	namespace modern_json = nlohmann;

	void ParseGNCPenaltySolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::GNCPenaltySolverSettings> gnc_penalty, const std::string string_end = "");

	void GenerateGNCPenaltySolver(modern_json::json& input, std::shared_ptr<optflow_solvers::GNCPenaltySolverSettings> gnc_penalty, const std::string string_end = "");
}