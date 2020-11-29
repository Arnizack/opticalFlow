#pragma once
#include"Hypodermic/Container.h"
#include"ContainerOptions.h"


namespace optflow_composition
{
	class ContainerInstaller
	{
	private:
		std::shared_ptr<ContainerOptions> _options = std::make_shared<ContainerOptions>();
	public:
		void SetOptions(std::shared_ptr<ContainerOptions> options);
		std::shared_ptr<Hypodermic::Container> Install();

	};
}