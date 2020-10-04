#include"pch.h"
#include "PyramidSolver.h"
namespace opticalflow_solvers
{
	namespace cs = core::solver;
	namespace csp = core::solver::problem;
	using PtrProblemTyp = std::shared_ptr<csp::IGrayPenaltyCrossProblem>;
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;

	PtrFlowField PyramidSolver::Solve(const PtrProblemTyp problem)
	{
		return PtrFlowField();
	}
	PtrFlowField PyramidSolver::Solve(const PtrProblemTyp problem, PtrFlowField initial_guess)
	{
		return PtrFlowField();
	}
}
