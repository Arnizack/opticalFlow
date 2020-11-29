#pragma once
#include"IncrementalSolverParser.h"

#ifndef VarName
#define VarName(var) (#var)
#endif // !VarName(var) (#var)


namespace json_settings
{
    void ParseIncrementalSolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::IncrementalSolverSettings> incremental_solver, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(incremental_solver->Steps), string_end);
        if (input.count(option_name))
            incremental_solver->Steps = input[option_name];
    }

    void GenerateIncrementalSolver(modern_json::json& input, std::shared_ptr<optflow_solvers::IncrementalSolverSettings> incremental_solver, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(incremental_solver->Steps), string_end);
        input[option_name] = incremental_solver->Steps;
    }
}