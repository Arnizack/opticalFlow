#pragma once
#include"CPUBackendInstaller.h"
#include"CPUBackendInstallers/RegisterCPUBackend.h"

namespace optflow_composition
{
    void CPUBackendInstaller::SetOptions(std::shared_ptr<CPUBackendOptions> options)
    {
        _options = options;
    }
    std::shared_ptr<Hypodermic::Container> CPUBackendInstaller::Install()
    {
        Hypodermic::ContainerBuilder builder;
        SetCPUBackendDefaultSettings(builder);
        RegisterCPUBackend(builder);
        return builder.build();
    }
}