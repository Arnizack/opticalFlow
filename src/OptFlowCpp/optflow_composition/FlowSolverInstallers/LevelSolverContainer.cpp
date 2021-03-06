#pragma once
#include"LevelSolverContainer.h"
#include"optflow_solvers/solvers/LinearizationSolver.h"
#include"optflow_solvers/solvers/IncrementalSolver.h"
#include"optflow_solvers/solvers/ConjugateGradientSolver.h"
#include"cpu_backend/linalg/ArithmeticChained.h"
#include"cpu_backend/linalg/ArithmeticVector.h"
#include"cpu_backend/ArrayFactory.h"

namespace optflow_composition
{
    std::shared_ptr<Hypodermic::Container> LevelSolverContainer(
        std::shared_ptr<Hypodermic::Container> backend,const LevelSettings& settings)
    {
        
        using ISolver = core::IFlowFieldSolver < std::shared_ptr<core::IGrayPenaltyCrossProblem>>;
        Hypodermic::ContainerBuilder builder;
        builder.registerType<optflow_solvers::LinearizationSolver>();
        builder.registerType<optflow_solvers::IncrementalSolver>()
            .with<ISolver, optflow_solvers::LinearizationSolver>();

        

        
        builder.registerInstance<optflow_solvers::LinearizationSolverSettings>(settings.Linearization);
        builder.registerInstance<optflow_solvers::IncrementalSolverSettings>(settings.Incremental);
       
        builder.registerInstance<optflow_solvers::CGSolverSettings>(settings.CGsolver);
        builder.registerType<optflow_solvers::ConjugateGradientSolver<double>>()
        .as<core::ILinearSolver<double>>();

        return builder.buildNestedContainerFrom(*backend);
    }



}