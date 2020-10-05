#pragma once
#include"core/IArray.h"
#include"core/IArrayFactory.h"
#include"core/solver/IFlowFieldSolver.h"
#include"core/IArrayFactory.h"

namespace opticalflow_solvers
{

	using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;

	class PyramidSolver : core::IFlowFieldSolver<PtrProblemTyp>
	{
	public:

		virtual PtrFlowField Solve(const PtrProblemTyp problem) override;

		virtual PtrFlowField Solve(
			const PtrProblemTyp problem, PtrFlowField initial_guess) override;
	
	private:
		std::shared_ptr<core::IArrayFactory<double, 3>> _flow_factory;

	};

}