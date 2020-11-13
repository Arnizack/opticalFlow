#pragma once
#include"pch.h"
#include"LinearizationSolver.h"
#include<math.h>

namespace optflow_solvers
{
    using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;
    using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

    LinearizationSolver::LinearizationSolver(
        std::shared_ptr<LinearizationSolverSettings> settings,
        std::shared_ptr<core::ICrossFlowFilter> cross_filter, 
        std::shared_ptr<ISunBakerLSUpdater> linear_system_updater, 
        PtrLinearSolver linear_solver, 
        std::shared_ptr<core::IReshaper<double>> flow_reshaper, 
        std::shared_ptr<core::IGrayWarper> warper, 
        std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory, 
        std::shared_ptr<core::IArithmeticBasic<double, 3>> flow_arithmetic)

        :
        _start_relaxation(settings->StartRelaxation),
        _end_relaxation(settings->EndRelaxation),
        _relaxation_steps(settings->RelaxationSteps),
        _cross_filter(cross_filter),
        _linear_system_updater(linear_system_updater),
        _linear_solver(linear_solver),
        _flow_reshaper(flow_reshaper),
        _warper(warper),
        _flow_factory(flow_factory),
        _flow_arithmetic(flow_arithmetic)
    {
    }

    PtrFlowField LinearizationSolver::Solve(const PtrProblemTyp problem)
    {
        size_t width = problem->FirstFrame->Shape[1];
        size_t height = problem->FirstFrame->Shape[0];
        PtrFlowField initial_guess = _flow_factory->Zeros({ 2,height,width });

        return Solve(problem, initial_guess);
    }
    PtrFlowField LinearizationSolver::Solve(const PtrProblemTyp problem, PtrFlowField initial_guess)
    {
        size_t width = problem->FirstFrame->Shape[1];
        size_t height = problem->FirstFrame->Shape[0];

        _warper->SetImage(problem->SecondFrame);

        auto warped_img = _warper->Warp(initial_guess);
        _linear_system_updater->SetFramePair(problem->FirstFrame, warped_img);
        _linear_system_updater->SetPenalty(problem->PenaltyFunc);

        auto delta_initial_flow = _flow_factory->Zeros({ 2,height,width });
        auto flow_before_filter = _flow_factory->Zeros({ 2,height,width });
        auto flow_after_filter = _flow_factory->Zeros({ 2,height,width });

        _cross_filter->SetCrossFilterImage(problem->CrossFilterImage);

        for (size_t relaxation_iter = 0; relaxation_iter < _relaxation_steps; relaxation_iter++)
        {
            double relaxation = ComputeRelaxation(relaxation_iter);
            _linear_system_updater->UpdateParameter(delta_initial_flow, relaxation);
            std::shared_ptr<core::ILinearProblem<double>> linear_problem 
                = _linear_system_updater->Update();

            using PtrVector = std::shared_ptr<core::IArray<double, 1>>;

            PtrVector guess_vector = _flow_reshaper->Reshape1D(delta_initial_flow);
            PtrVector result_vector = _linear_solver->Solve(linear_problem, guess_vector);
            
            delta_initial_flow = _flow_reshaper->Reshape3D(result_vector, {2,height,width});

             _flow_arithmetic->AddTo(flow_before_filter, initial_guess, delta_initial_flow);
             _cross_filter->SetAuxiliaryInfluence(relaxation);
             _cross_filter->ApplyTo(flow_after_filter, flow_before_filter);

             _flow_arithmetic->SubTo(delta_initial_flow, flow_after_filter, initial_guess);
        }
        return flow_after_filter;
    }
    double LinearizationSolver::ComputeRelaxation(size_t relaxation_iter)
    {
        if(_relaxation_steps >1)
        {
            double exponent = exp(_start_relaxation);
            double bruch_oben =  exp(_end_relaxation) - exp(_start_relaxation);
            double bruch_unten = _relaxation_steps - 1;
            double x = relaxation_iter;

            exponent += (bruch_oben / bruch_unten) * x;

            return log(exponent);
        }
        return _start_relaxation;
    }
}