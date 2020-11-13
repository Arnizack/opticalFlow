#pragma once
#include"RegisterSunBakerSolver.h"
#include"optflow_solvers/solvers/LinearizationSolver.h"
#include"optflow_solvers/solvers/IncrementalSolver.h"
#include"optflow_solvers/solvers/PyramidSolver.h"
#include"optflow_solvers/solvers/GNCPenaltySolver.h"
#include"optflow_solvers/solvers/PyramidSolverIterator.h"
#include"optflow_solvers/pyramid/GrayPenaltyCrossPyramidBuilder.h"

namespace console_ui
{
    using IFlowSolver = core::IFlowFieldSolver<
        std::shared_ptr<core::IGrayPenaltyCrossProblem>>;
    void RegisterSunBakerSolver(Hypodermic::ContainerBuilder& builder)
    {
        _RegisterLevelSolver(builder);
        _RegisterPyramidSolver(builder);
        _RegisterGNCSolver(builder);
        
    }
    void _RegisterLevelSolver(Hypodermic::ContainerBuilder& builder)
    {
        builder.registerType<optflow_solvers::LinearizationSolver>();
        builder.registerType<optflow_solvers::IncrementalSolver>()
            .with<IFlowSolver, optflow_solvers::LinearizationSolver>();
        

        

    }

    void _RegisterGNCSolver(Hypodermic::ContainerBuilder& builder)
    {
        
        builder.registerType<optflow_solvers::GNCPenaltySolver>()
            .as < core::IFlowFieldSolver<std::shared_ptr<core::IGrayCrossFilterProblem>>>();
    }

    void _RegisterPyramidSolver(Hypodermic::ContainerBuilder& builder)
    {  
        builder.registerType<optflow_solvers::GrayPenaltyCrossPyramidBuilder>()
            .as<core::IPyramidBuilder<core::IGrayPenaltyCrossProblem>>();

        builder.registerType<optflow_solvers::PyramidSolver>()
            .with<IFlowSolver, optflow_solvers::IncrementalSolver>();
        
        using ISolverIterator = core::IFlowSolverIterator<core::IGrayPenaltyCrossProblem>;

        builder.registerType<optflow_solvers::PyramidSolverIterator>()
            .as< core::IFlowSolverIterator<core::IGrayPenaltyCrossProblem>>();


    }

    void SetDefaultSunBakerSettings(Hypodermic::ContainerBuilder& builder)
    {
        auto linearization_settings = std::make_shared< optflow_solvers::LinearizationSolverSettings>();
        builder.registerInstance<optflow_solvers::LinearizationSolverSettings>(linearization_settings);
        
        auto incremental_settings = std::make_shared<optflow_solvers::IncrementalSolverSettings>();
        builder.registerInstance<optflow_solvers::IncrementalSolverSettings>(incremental_settings);

        auto gnc_settings = std::make_shared<optflow_solvers::GNCPenaltySolverSettings>();
        builder.registerInstance<optflow_solvers::GNCPenaltySolverSettings>(gnc_settings);

        auto pyramids_resolutions = std::make_shared< optflow_solvers::PyramidsResolutions>();
        pyramids_resolutions->Resolutions = std::vector<optflow_solvers::PyramidResolutions>(2);
        pyramids_resolutions->Resolutions[0].MinResolutionX = 32;
        pyramids_resolutions->Resolutions[0].MinResolutionY= 32;
        pyramids_resolutions->Resolutions[0].ScaleFactor = 0.5;

        pyramids_resolutions->Resolutions[1].MinResolutionX = 200;
        pyramids_resolutions->Resolutions[1].MinResolutionY = 200;
        pyramids_resolutions->Resolutions[1].ScaleFactor = 0.5;

        builder.registerInstance<optflow_solvers::PyramidsResolutions>(pyramids_resolutions);



    }
}