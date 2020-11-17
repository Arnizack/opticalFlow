#pragma once
#include"Hypodermic/ContainerBuilder.h"

#include <boost/program_options.hpp>

namespace optflow_composition
{
	void RegisterCPUBackend(Hypodermic::ContainerBuilder& builder);
	void SetCPUBackendDefaultSettings(Hypodermic::ContainerBuilder& builder);
	void SetCPUBackendCommandlineSettings(Hypodermic::ContainerBuilder& builder, boost::program_options::variables_map vm);
}