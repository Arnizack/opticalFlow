#pragma once
#include"core/IArray.h"
#include"core/IArrayFactory.h"
#include"core/solver/IFlowFieldSolver.h"
#include"core/IArrayFactory.h"

namespace opticalflow_solvers
{
	namespace cs = core::solver;
	namespace csp = core::solver::problem;
	using PtrProblemTyp = std::shared_ptr<csp::IGrayPenaltyCrossProblem>;
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;

	class PyramidSolver : cs::IFlowFieldSolver<PtrProblemTyp>
	{
	public:



		// Inherited via IFlowFieldSolver
		virtual PtrFlowField Solve(const PtrProblemTyp problem) override;
		virtual PtrFlowField Solve(const PtrProblemTyp problem, PtrFlowField initial_guess) override;
	
	private:
		std::shared_ptr<core::IArrayFactory<double, 3>> _flow_factory;

	};

}