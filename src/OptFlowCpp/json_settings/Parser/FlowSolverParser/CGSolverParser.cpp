#pragma once
#include "CGSolverParser.h"

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName(var) (#var)


namespace json_settings
{
    void ParseCGSolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::CGSolverSettings> cg_solver, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(cg_solver->Iterations), string_end);
        if (input.count(option_name))
            cg_solver->Iterations = input[option_name];

        option_name = GetOptionName(VarName(cg_solver->Tolerance), string_end);
        if (input.count(option_name))
            cg_solver->Tolerance = input[option_name];
    }

    void GenerateCGSolver(modern_json::json& output, std::shared_ptr<optflow_solvers::CGSolverSettings> cg_solver, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(cg_solver->Iterations), string_end);
        output[option_name] = cg_solver->Iterations;

        option_name = GetOptionName(VarName(cg_solver->Tolerance), string_end);
        output[option_name] = cg_solver->Tolerance;
    }
}