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
	void RegisterDefaultCGSettings(Hypodermic::ContainerBuilder& builder)
	{
		auto cg_settings = std::make_shared<optflow_solvers::CGSolverSettings>();
		builder.registerInstance<optflow_solvers::CGSolverSettings>(cg_settings);
	}
}