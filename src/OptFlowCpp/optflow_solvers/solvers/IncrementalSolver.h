#pragma once
#include"core/solver/IFlowFieldSolver.h"
#include"core/IArray.h"
#include"core/IArrayFactory.h"

namespace optflow_solvers
{
	struct IncrementalSolverSettings
	{
		int Steps = 3;
	};
	using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;
	class IncrementalSolver : public core::IFlowFieldSolver<PtrProblemTyp>
	{
	public:
		using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;
		using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

		IncrementalSolver(std::shared_ptr<IncrementalSolverSettings> settings,
			std::shared_ptr<core::IFlowFieldSolver<PtrProblemTyp>> inner_solver,
			std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory);

		// Inherited via IFlowFieldSolver
		virtual PtrFlowField Solve(const PtrProblemTyp problem) override;

		virtual PtrFlowField Solve(const PtrProblemTyp problem, PtrFlowField initial_guess) override;

	private:
		std::shared_ptr<core::IFlowFieldSolver<PtrProblemTyp>> _inner_solver;
		std::shared_ptr<core::IArrayFactory<double, 3>> _flow_factory;
		int _steps;
	};
}