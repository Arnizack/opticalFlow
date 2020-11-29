#pragma once
#include "Hypodermic/Container.h"
#include "Hypodermic/ContainerBuilder.h"

#include "RegisterCPULinalg.h"
#include "RegisterCPUBackend.h"
#include "../CPUBackendOptions.h"
#include"optflow_solvers/linearsystems/ISunBakerLSUpdater.h"

namespace optflow_composition
{
	std::shared_ptr<Hypodermic::Container> CPUBackendContainer(const std::shared_ptr<CPUBackendOptions> settings);
}