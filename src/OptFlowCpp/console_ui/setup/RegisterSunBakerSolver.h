#pragma once
#include"Hypodermic/ContainerBuilder.h"
#include <boost/program_options.hpp>

namespace console_ui
{
	void RegisterSunBakerSolver(Hypodermic::ContainerBuilder& builder);
	void _RegisterLevelSolver(Hypodermic::ContainerBuilder& builder);
	void _RegisterGNCSolver(Hypodermic::ContainerBuilder& builder);
	void _RegisterPyramidSolver(Hypodermic::ContainerBuilder& builder);
	void SetDefaultSunBakerSettings(Hypodermic::ContainerBuilder& builder);
	void SetCommandlineSunBakerSettings(Hypodermic::ContainerBuilder& builder, boost::program_options::variables_map vm);
}