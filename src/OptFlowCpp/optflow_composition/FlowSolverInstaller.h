#pragma once
#include"FlowSolverOptions.h"
namespace optflow_composition
{
	class FlowSolverInstaller
	{
	private:
		std::shared_ptr<FlowSolverOptions> _options = std::make_shared<FlowSolverOptions>();
	public:
		void SetOptions(std::shared_ptr<FlowSolverOptions> options);
		std::shared_ptr<Hypodermic::Container> Install(const Backends& backends);
	};
}