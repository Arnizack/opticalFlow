#pragma once

#include"CPUBackendOptions.h"
#include"FlowSolverOptions.h"
namespace optflow_composition
{
	struct ContainerOptions
	{
		std::shared_ptr<CPUBackendOptions> CPUOptions = std::make_shared<CPUBackendOptions>();
		std::shared_ptr<FlowSolverOptions> SolverOptions = std::make_shared<FlowSolverOptions>();
	};
}