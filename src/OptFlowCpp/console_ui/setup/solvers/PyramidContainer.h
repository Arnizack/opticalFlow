#pragma once
#include"Hypodermic/ContainerBuilder.h"
#include"SetupSolverSettings.h"

namespace console_ui
{

	std::shared_ptr<Hypodermic::Container> PyramidContainer(const Backends& backends, const PyramidSettings& settings);
}