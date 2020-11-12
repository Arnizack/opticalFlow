#pragma once
#include"Hypodermic/ContainerBuilder.h"

namespace console_ui
{
	void RegisterSunBakerSolver(Hypodermic::ContainerBuilder& builder);
	void _RegisterLevelSolver(Hypodermic::ContainerBuilder& builder);
	void _RegisterGNCSolver(Hypodermic::ContainerBuilder& builder);
	void _RegisterPyramidSolver(Hypodermic::ContainerBuilder& builder);
	void SetDefaultSunBakerSettings(Hypodermic::ContainerBuilder& builder);
}