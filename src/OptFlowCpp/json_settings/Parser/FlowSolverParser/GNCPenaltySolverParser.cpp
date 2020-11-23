#pragma once
#include"GNCPenaltySolverParser.h"

#ifndef VarName
#define VarName(var) (#var)
#endif // !VarName


namespace json_settings
{
    void ParseGNCPenaltySolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::GNCPenaltySolverSettings> gnc_penalty, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(gnc_penalty->GNCSteps), string_end);
        if (input.count(option_name))
            gnc_penalty->GNCSteps = input[option_name];
    }

    void GenerateGNCPenaltySolver(modern_json::json& input, std::shared_ptr<optflow_solvers::GNCPenaltySolverSettings> gnc_penalty, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(gnc_penalty->GNCSteps), string_end);
        input[option_name] = gnc_penalty->GNCSteps;
    }
}