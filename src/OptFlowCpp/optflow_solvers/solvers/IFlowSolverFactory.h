#pragma once
#include"core/solver/IFlowFieldSolver.h"
#include"../framework.h"
namespace optflow_solvers
{

	class IFlowSolverFactory
	{
	public:
		
		using PtrStandardFlowSolver = 
			std::shared_ptr<core::IFlowFieldSolver<
			std::shared_ptr<core::IGrayPenaltyCrossProblem>>>;

		using PtrPenaltyFlowSolver = 
			std::shared_ptr<core::IFlowFieldSolver<
			std::shared_ptr<core::IGrayCrossFilterProblem>>>;

		virtual PtrPenaltyFlowSolver CreateGNCPenaltySolver() = 0;

		virtual PtrStandardFlowSolver CreatePyramidSolver() = 0;

		virtual PtrStandardFlowSolver CreateIncreamentalSolver() = 0;

		virtual PtrStandardFlowSolver CreateRelaxationFilterSolver() = 0;

		
	};
}