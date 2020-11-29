#pragma once
#include"Hypodermic/ContainerBuilder.h"


namespace optflow_composition
{
	void RegisterCPUBackend(Hypodermic::ContainerBuilder& builder);
	void SetCPUBackendDefaultSettings(Hypodermic::ContainerBuilder& builder);
}