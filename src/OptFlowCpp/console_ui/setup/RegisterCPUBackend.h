#pragma once
#include"Hypodermic/ContainerBuilder.h"

namespace console_ui
{
	void RegisterCPUBackend(Hypodermic::ContainerBuilder& builder);
	void _RegisterCPULinalg(Hypodermic::ContainerBuilder& builder);
	void SetCPUBackendDefaultSettings(Hypodermic::ContainerBuilder& builder);
	
}