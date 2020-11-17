#pragma once
#include"Hypodermic/Container.h"
#include"../FlowSolverOptions.h"

namespace optflow_composition
{
	

	std::shared_ptr<Hypodermic::Container> GNCSolverContainer(const Backends& backends, const GNCSolverSettings& settings);
}