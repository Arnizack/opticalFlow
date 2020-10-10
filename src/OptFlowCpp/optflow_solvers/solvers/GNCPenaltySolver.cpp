#include "pch.h"
#include "GNCPenaltySolver.h"
namespace optflow_solvers
{
    using ProblemTyp = std::shared_ptr<core::IGrayCrossFilterProblem>;
    
    using PtrFlowField = std::shared_ptr < core::IArray<double, 3> >;
    
    using PtrStandardFlowSolver = 
        std::shared_ptr<core::IFlowFieldSolver<
        std::shared_ptr<core::IGrayPenaltyCrossProblem>>>;
    
    using PtrBlendPenalty = std::shared_ptr<
        core::IBlendablePenalty<core::IArray<float, 2>>>;
    
    using PtrFlowFactory = std::shared_ptr<core::IArrayFactory<double, 3>>;

    using PtrProblemFactory = std::shared_ptr<core::IProblemFactory>;

    using PtrPenaltyProblem = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

    GNCPenaltySolver::GNCPenaltySolver(int gnc_steps, 
        std::vector<PtrStandardFlowSolver> inner_solvers, 
        PtrBlendPenalty penalty_func, PtrFlowFactory flow_factory, 
        PtrProblemFactory problem_factory)
        : _gnc_steps(gnc_steps), _inner_solvers(inner_solvers), 
        _flow_factory(flow_factory), _penalty_func(penalty_func),
        _problem_factory(problem_factory)
    {
    }

    PtrFlowField GNCPenaltySolver::Solve(const ProblemTyp problem)
    {
        
        size_t width = problem->FirstFrame->Shape[1];
        size_t height = problem->FirstFrame->Shape[0];
        PtrFlowField initial_guess = _flow_factory->Zeros({2,height,width});

        return Solve(problem, initial_guess);
    }
    PtrFlowField GNCPenaltySolver::Solve(const ProblemTyp problem, PtrFlowField initial_guess)
    {
        PtrPenaltyProblem penalty_problem = _problem_factory->CreateGrayPenaltyCrossProblem();
        penalty_problem->CrossFilterImage = problem->CrossFilterImage;
        penalty_problem->FirstFrame = problem->FirstFrame;
        penalty_problem->SecondFrame = problem->SecondFrame;



        for (int gnc_iter = 0; gnc_iter < _gnc_steps; gnc_iter++)
        {
            double blend_factor = ComputeBlendFactor(gnc_iter,_gnc_steps);
            _penalty_func->SetBlendFactor(blend_factor);
            penalty_problem->PenaltyFunc = _penalty_func;
            PtrStandardFlowSolver current_solver = GetFlowSolverAt(gnc_iter);
            
            initial_guess = current_solver->Solve(penalty_problem, initial_guess);
        }
        return initial_guess;
        
    }
    double GNCPenaltySolver::ComputeBlendFactor(int gnc_iter, int gnc_steps)
    {
        if (gnc_steps > 1)
            return (double)gnc_iter / (double)(gnc_steps - 1);
        
        return 0;
            
    }
    PtrStandardFlowSolver GNCPenaltySolver::GetFlowSolverAt(int gnc_iter)
    {
        size_t length = _inner_solvers.size();
        if (gnc_iter < length)
        {
            return _inner_solvers[gnc_iter];
        }
        return _inner_solvers[length - 1];
    }
}