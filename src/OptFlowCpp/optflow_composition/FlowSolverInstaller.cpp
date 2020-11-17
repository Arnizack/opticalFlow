#pragma once
#include"FlowSolverInstaller.h"
#include"FlowSolverInstallers/GNCSolverContainer.h"

namespace optflow_composition
{
	void FlowSolverInstaller::SetOptions(std::shared_ptr<FlowSolverOptions> options)
	{
		_options = options;
	}
	std::shared_ptr<Hypodermic::Container> FlowSolverInstaller::Install(const Backends& backends)
	{
		return GNCSolverContainer(backends, _options->GNCSettings);
	}
}