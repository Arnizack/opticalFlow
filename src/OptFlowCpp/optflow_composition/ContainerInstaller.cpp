#pragma once
#include"ContainerInstaller.h"
#include"FlowSolverInstaller.h"
#include"Backends.h"
#include"CPUBackendInstaller.h"
#include"Hypodermic/ContainerBuilder.h"
#include"optflow_solvers/OpticalFlowApplication.h"
namespace optflow_composition
{
    void ContainerInstaller::SetOptions(std::shared_ptr<ContainerOptions> options)
    {
        _options = options;
    }
    std::shared_ptr<Hypodermic::Container> ContainerInstaller::Install()
    {
        
        CPUBackendInstaller cpu_installer;
        cpu_installer.SetOptions(_options->CPUOptions);
        auto cpu_container = cpu_installer.Install();
        Backends backends(cpu_container);
        FlowSolverInstaller flow_installer;
        flow_installer.SetOptions(_options->SolverOptions);
        auto flow_container = flow_installer.Install(backends);
        Hypodermic::ContainerBuilder builder;
        builder.registerType<optflow_solvers::OpticalFlowApplication>();
        return builder.buildNestedContainerFrom(*flow_container);
    }
}