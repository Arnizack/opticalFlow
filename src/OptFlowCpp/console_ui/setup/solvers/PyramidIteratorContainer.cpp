#pragma once
#include"PyramidIteratorContainer.h"
#include"PyramidContainer.h"
#include"optflow_solvers/solvers/TwoSolverIterator.h"
#include"LevelSolverContainer.h"

namespace console_ui
{

    std::shared_ptr<Hypodermic::Container> PyramidIteratorContainer(
        const Backends& backends,
        const PyramidIteratorSettings& settings_container)
    {
        Hypodermic::ContainerBuilder builder;

        auto convex_container = PyramidContainer(backends,settings_container.ConvexSettings);
        auto nonconvex_container = PyramidContainer(backends, settings_container.NonConvexSettings);

        using ISolver = core::IFlowFieldSolver < std::shared_ptr<core::IGrayPenaltyCrossProblem>>;

        auto convex_solver = convex_container->resolve<ISolver>();
        auto nonconvex_solver = nonconvex_container->resolve<ISolver>();

        auto iterator = std::make_shared< optflow_solvers::TwoSolverIterator<core::IGrayPenaltyCrossProblem>>(
            convex_solver, nonconvex_solver
            );

        builder.registerInstance<optflow_solvers::TwoSolverIterator<core::IGrayPenaltyCrossProblem>>(iterator)
            .as<core::IFlowSolverIterator<core::IGrayPenaltyCrossProblem>>();

        return builder.build();
       
    }
    
}