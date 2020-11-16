#pragma once
#include"Hypodermic/Container.h"
#include"SetupSolverSettings.h"

namespace console_ui
{
	

	std::shared_ptr<Hypodermic::Container> GNCSolverContainer(const Backends& backends, const GNCSolverSettings& settings);
}