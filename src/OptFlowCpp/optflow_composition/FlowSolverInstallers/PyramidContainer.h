#pragma once
#include"Hypodermic/ContainerBuilder.h"
#include"../FlowSolverOptions.h"

namespace optflow_composition
{

	std::shared_ptr<Hypodermic::Container> PyramidContainer(const Backends& backends, const PyramidSettings& settings);
}