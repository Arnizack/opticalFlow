#pragma once
#include"LinearizationSolverParser.h"

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif

namespace console_ui
{
	void ParseLinearizationSolver(const modern_json::json& input, std::shared_ptr<optflow_solvers::LinearizationSolverSettings> linearization_solver, const std::string string_end)
	{
		std::string option_name = GetOptionName(VarName(linearization_solver->EndRelaxation), string_end);
		if (input.count(option_name))
			linearization_solver->EndRelaxation = input[option_name];

		option_name = GetOptionName(VarName(linearization_solver->StartRelaxation), string_end);
		if (input.count(option_name))
			linearization_solver->StartRelaxation = input[option_name];

		option_name = GetOptionName(VarName(linearization_solver->RelaxationSteps), string_end);
		if (input.count(option_name))
			linearization_solver->RelaxationSteps = input[option_name];
	}

	void GenerateLinearizationSolver(modern_json::json& input, std::shared_ptr<optflow_solvers::LinearizationSolverSettings> linearization_solver, const std::string string_end)
	{
		std::string option_name = GetOptionName(VarName(linearization_solver->EndRelaxation), string_end);
		input[option_name] = linearization_solver->EndRelaxation;

		option_name = GetOptionName(VarName(linearization_solver->StartRelaxation), string_end);
		input[option_name] = linearization_solver->StartRelaxation;

		option_name = GetOptionName(VarName(linearization_solver->RelaxationSteps), string_end);
		input[option_name] = linearization_solver->RelaxationSteps;
	}
}