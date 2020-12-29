#pragma once
#include "Hypodermic/Container.h"

namespace optflow_composition
{
	struct Backends
	{
		std::shared_ptr<Hypodermic::Container> CPUBackend;
		Backends(std::shared_ptr<Hypodermic::Container> cpu_backend)
			: CPUBackend(cpu_backend)
		{}

	};
}