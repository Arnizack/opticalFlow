#pragma once
#include"pch.h"
#include"LinearizationSolver.h"
#include<math.h>

namespace optflow_solvers
{
    using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;
    using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;
    PtrFlowField LinearizationSolver::Solve(const PtrProblemTyp problem)
    {
        return PtrFlowField();
    }
    PtrFlowField LinearizationSolver::Solve(const PtrProblemTyp problem, PtrFlowField initial_guess)
    {
        _linear_system_builder->SetFramePair(problem->FirstFrame, problem->SecondFrame);
        for (size_t relaxation_iter = 0; relaxation_iter < _relaxation_steps; relaxation_iter++)
        {
            double relaxation = ComputeRelaxation(relaxation_iter);
            _linear_system_builder->UpdateParameter(initial_guess, relaxation);
            auto linear_problem = _linear_system_builder->Create();
            
        }
        return nullptr;
    }
    double LinearizationSolver::ComputeRelaxation(size_t relaxation_iter)
    {
        double exponent = exp(_start_relaxation);
        double bruch_oben =  exp(_end_relaxation) - exp(_start_relaxation);
        double bruch_unten = _relaxation_steps - 1;
        double x = relaxation_iter;

        exponent += (bruch_oben / bruch_unten) * x;

        return log(exponent);
    }
}