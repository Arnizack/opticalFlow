#pragma once
#include"PyramidContainer.h"
#include"LevelSolverContainer.h"
#include"optflow_solvers/solvers/PyramidSolver.h"
#include"optflow_solvers/pyramid/GrayPenaltyCrossPyramidBuilder.h"
#include"optflow_solvers/solvers/LinearizationSolver.h"
#include"optflow_solvers/solvers/IncrementalSolver.h"
#include"../RegisterCPUBackend.h"

namespace console_ui
{
    std::shared_ptr<Hypodermic::Container> PyramidContainer(const Backends& backends, const PyramidSettings& settings)
    {
        using ISolver = core::IFlowFieldSolver<std::shared_ptr<core::IGrayPenaltyCrossProblem>>;
        Hypodermic::ContainerBuilder builder;

        auto cpu_backend = backends.CPUBackend;

        auto pyramid_builder = std::make_shared< optflow_solvers::GrayPenaltyCrossPyramidBuilder>
            (cpu_backend->resolve< core::IScaler<core::IGrayPenaltyCrossProblem>>());

        pyramid_builder->SetScaleFactor(settings.Resolution.ScaleFactor, 
            std::array<size_t, 2>{settings.Resolution.MinResolutionX, settings.Resolution.MinResolutionY});

        builder.registerInstance < optflow_solvers::GrayPenaltyCrossPyramidBuilder>(pyramid_builder)
            .as < core::IPyramidBuilder<core::IGrayPenaltyCrossProblem>>();

        auto single_level_container = LevelSolverContainer(cpu_backend,settings.SingleLevelSettings);
        auto single_level_solver = single_level_container->resolve<optflow_solvers::IncrementalSolver>();
        builder.registerInstance< optflow_solvers::IncrementalSolver>(single_level_solver);
        builder.registerType<optflow_solvers::PyramidSolver>()
            .with<ISolver, optflow_solvers::IncrementalSolver>()
            .as<ISolver>();
        return builder.buildNestedContainerFrom(*cpu_backend);
    }



}