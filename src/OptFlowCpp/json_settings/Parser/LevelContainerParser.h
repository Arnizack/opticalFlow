#pragma once

#include <nlohmann/json.hpp>

#include "optflow_composition/ContainerOptions.h"

#include "FlowSolverParser/CGSolverParser.h"
#include "FlowSolverParser/IncrementalSolverParser.h"
#include "FlowSolverParser/LinearizationSolverParser.h"

namespace json_settings
{
	namespace modern_json = nlohmann;

	void ParseLevelSettings(const modern_json::json& input, optflow_composition::LevelSettings& level, std::string string_end = "");

	void GenerateLevelSettings(modern_json::json& output, optflow_composition::LevelSettings& level, std::string string_end = "");
}