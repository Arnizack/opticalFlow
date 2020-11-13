#pragma once
#include"Hypodermic/ContainerBuilder.h"

namespace console_ui
{
	void RegisterCGSolver(Hypodermic::ContainerBuilder& builder);
	void SetDefaultCGSettings(Hypodermic::ContainerBuilder& builder);
}