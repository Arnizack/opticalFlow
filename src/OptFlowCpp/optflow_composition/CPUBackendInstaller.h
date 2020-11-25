#pragma once
#include"CPUBackendOptions.h"

#include"Hypodermic/Container.h"
#include "CPUBackendInstallers/CPUBackendContainer.h"

namespace optflow_composition
{
	class CPUBackendInstaller
	{
	private:
		std::shared_ptr<CPUBackendOptions> _options = std::make_shared<CPUBackendOptions>();
	public:
		void SetOptions(std::shared_ptr<CPUBackendOptions> options);
		std::shared_ptr<Hypodermic::Container> Install();
	};
}