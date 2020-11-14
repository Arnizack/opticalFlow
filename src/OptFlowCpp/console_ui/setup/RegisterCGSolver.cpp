#pragma once
#include"RegisterCGSolver.h"
#include"optflow_solvers/solvers/ConjugateGradientSolver.h"

namespace console_ui
{
	void RegisterCGSolver(Hypodermic::ContainerBuilder& builder)
	{
		builder.registerType<optflow_solvers::ConjugateGradientSolver<double>>()
			.as<core::ILinearSolver<double>>();

	}
	void SetDefaultCGSettings(Hypodermic::ContainerBuilder& builder)
	{
		auto cg_settings = std::make_shared<optflow_solvers::CGSolverSettings>();
		builder.registerInstance<optflow_solvers::CGSolverSettings>(cg_settings);
	}

	void SetCommandlineCGSettings(Hypodermic::ContainerBuilder& builder, boost::program_options::variables_map vm)
	{
		auto cg_settings = std::make_shared<optflow_solvers::CGSolverSettings>();
		cg_settings->Iterations = vm["cg_iter"].as<size_t>();
		cg_settings->Tolerance = vm["cg_tol"].as<double>();
		builder.registerInstance<optflow_solvers::CGSolverSettings>(cg_settings);
	}
}