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

		virtual PtrPenaltyFlowSolver CreateGNCPenaltySolver() = 0;

		virtual PtrStandardFlowSolver CreatePyramidSolver() = 0;

		virtual PtrStandardFlowSolver CreateIncreamentalSolver() = 0;

		virtual PtrStandardFlowSolver CreateRelaxationFilterSolver() = 0;

		
	};
}