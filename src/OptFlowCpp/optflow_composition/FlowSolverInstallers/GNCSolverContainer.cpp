#pragma once
#include"GNCSolverContainer.h"
#include"PyramidIteratorContainer.h"
#include"optflow_solvers/solvers/GNCPenaltySolver.h"
namespace optflow_composition
{
    std::shared_ptr<Hypodermic::Container> GNCSolverContainer(const Backends& backends, const GNCSolverSettings& settings)
    {
        Hypodermic::ContainerBuilder builder;
        builder.registerType<optflow_solvers::GNCPenaltySolver>()
            .as<core::IFlowFieldSolver<std::shared_ptr<core::IGrayCrossFilterProblem>>>();
        builder.registerInstance<optflow_solvers::GNCPenaltySolverSettings>(settings.GNCSettings);

        using IIteratorSolver = core::IFlowSolverIterator<core::IGrayPenaltyCrossProblem>;
        auto pyramid_iterator_container = PyramidIteratorContainer(backends,
            settings.PyramidContainerSettings);
        auto iterator_solver = pyramid_iterator_container->resolve<core::IFlowSolverIterator<core::IGrayPenaltyCrossProblem>>();

        builder.registerInstance<core::IFlowSolverIterator<core::IGrayPenaltyCrossProblem>>(iterator_solver);
        return builder.buildNestedContainerFrom(*backends.CPUBackend);
    }

}