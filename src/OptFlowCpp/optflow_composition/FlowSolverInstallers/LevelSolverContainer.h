#pragma once
#include"Hypodermic/ContainerBuilder.h"
#include"../FlowSolverOptions.h"


namespace optflow_composition
{
	
	std::shared_ptr<Hypodermic::Container>
		LevelSolverContainer(std::shared_ptr<Hypodermic::Container> backend, 
			const LevelSettings& settings);
}