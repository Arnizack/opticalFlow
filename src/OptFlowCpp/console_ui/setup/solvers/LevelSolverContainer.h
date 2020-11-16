#pragma once
#include"Hypodermic/ContainerBuilder.h"
#include"SetupSolverSettings.h"


namespace console_ui
{
	
	std::shared_ptr<Hypodermic::Container>
		LevelSolverContainer(std::shared_ptr<Hypodermic::Container> backend, 
			const LevelSettings& settings);
}