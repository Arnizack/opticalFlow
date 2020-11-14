#pragma once
#include"pch.h"
#include"IncrementalSolver.h"
#include <iostream>

namespace optflow_solvers
{
    using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;
    using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

    IncrementalSolver::IncrementalSolver(std::shared_ptr<IncrementalSolverSettings> settings,
        std::shared_ptr<core::IFlowFieldSolver<PtrProblemTyp>> inner_solver, 
        std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory)
        :_inner_solver(inner_solver),_flow_factory(flow_factory),_steps(settings->Steps)
    {
    }

    PtrFlowField IncrementalSolver::Solve(const PtrProblemTyp problem)
    {
        size_t width = problem->FirstFrame->Shape[1];
        size_t height = problem->FirstFrame->Shape[0];
        PtrFlowField initial_guess = _flow_factory->Zeros({ 2,height,width });

        return Solve(problem, initial_guess);
    }
    PtrFlowField IncrementalSolver::Solve(const PtrProblemTyp problem, PtrFlowField initial_guess)
    {
        for (int step = 0; step < _steps; step++)
        {
            std::cout << "Incremental Solver Step: " << step << std::endl;
            initial_guess = _inner_solver->Solve(problem, initial_guess);
        }
        return initial_guess;
    }
}