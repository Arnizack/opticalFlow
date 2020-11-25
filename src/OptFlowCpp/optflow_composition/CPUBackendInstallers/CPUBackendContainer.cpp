#pragma once
#include"CPUBackendContainer.h"

namespace optflow_composition
{
	std::shared_ptr<Hypodermic::Container> CPUBackendContainer(const std::shared_ptr<CPUBackendOptions> settings)
	{
		Hypodermic::ContainerBuilder builder;

		RegisterCPULinalg(builder);

		RegisterCPUBackend(builder);

		builder.registerInstance<cpu_backend::CharbonnierPenaltySettings>(settings->CharbonnierPenalty);

		builder.registerInstance<cpu_backend::CrossMedianFilterSettings>(settings->CrossMedianFilter);

		builder.registerInstance<cpu_backend::LinearSystemSettings>(settings->LinearSystem);

		return builder.build();
	}
}