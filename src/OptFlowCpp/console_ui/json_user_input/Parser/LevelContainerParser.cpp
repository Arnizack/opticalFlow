#pragma once
#include"LevelContainerParser.h"

namespace console_ui
{
    void ParseLevelSettings(const modern_json::json& input, optflow_composition::LevelSettings& level, std::string string_end)
    {
        ParseCGSolver(input, level.CGsolver, string_end);
        ParseIncrementalSolver(input, level.Incremental, string_end);
        ParseLinearizationSolver(input, level.Linearization, string_end);
    }

    void GenerateLevelSettings(modern_json::json& output, optflow_composition::LevelSettings& level, std::string string_end)
    {
        GenerateCGSolver(output, level.CGsolver, string_end);
        GenerateIncrementalSolver(output, level.Incremental, string_end);
        GenerateLinearizationSolver(output, level.Linearization, string_end);
    }
}