#pragma once
#include "Hypodermic/ContainerBuilder.h"
#include <boost/program_options.hpp>

namespace console_ui
{
	void RegisterCGSolver(Hypodermic::ContainerBuilder& builder);
	void SetDefaultCGSettings(Hypodermic::ContainerBuilder& builder);
	void SetCommandlineCGSettings(Hypodermic::ContainerBuilder& builder, boost::program_options::variables_map vm);
}