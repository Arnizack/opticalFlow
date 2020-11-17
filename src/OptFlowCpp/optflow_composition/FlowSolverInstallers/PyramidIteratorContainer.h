#pragma once

#include"Hypodermic/ContainerBuilder.h"
#include"../FlowSolverOptions.h"

namespace optflow_composition
{

	std::shared_ptr<Hypodermic::Container> PyramidIteratorContainer(
		const Backends& backends,
		const PyramidIteratorSettings& settings_container);

}