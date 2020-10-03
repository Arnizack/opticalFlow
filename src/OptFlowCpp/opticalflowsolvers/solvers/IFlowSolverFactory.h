#pragma once
#include"core/solver/IFlowFieldSolver.h"
#include"../framework.h"
namespace opticalflow_solvers
{
	namespace cs = core::solver;
	namespace csp = core::solver::problem;

	class IFlowSolverFactory
	{
	public:
		
		using PtrStandardFlowSolver = std::shared_ptr<cs::IFlowFieldSolver<std::shared_ptr<csp::IGrayPenaltyCrossProblem>>>;

		using PtrPenaltyFlowSolver = std::shared_ptr<cs::IFlowFieldSolver<std::shared_ptr<csp::IGrayCrossFilterProblem>>>;

		virtual PtrPenaltyFlowSolver CreateGNCPenaltySolver(int gnc_steps, std::vector<PtrStandardFlowSolver> inner_solvers) = 0;

		virtual PtrStandardFlowSolver CreatePyramidSolver(PtrStandardFlowSolver inner_solver) = 0;

		virtual PtrStandardFlowSolver CreateIncreamentalSolver(PtrStandardFlowSolver inner_solver) = 0;

		virtual PtrStandardFlowSolver CreateRelaxationFilterSolver() = 0;

		
	};
}