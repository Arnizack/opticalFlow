#pragma once

#include"Hypodermic/ContainerBuilder.h"
#include"SetupSolverSettings.h"

namespace console_ui
{

	std::shared_ptr<Hypodermic::Container> PyramidIteratorContainer(
		const Backends& backends,
		const PyramidIteratorSettings& settings_container);

}