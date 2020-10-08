#pragma once
#include"core/solver/IFlowFieldSolver.h"
#include"core/solver/problem/IGrayCrossFilterProblem.h"
#include"core/flow/ICrossFlowFilter.h"
#include"core/solver/ILinearSolver.h"
#include"../linearsystems/ISunBakerLSBuilder.h"

namespace optflow_solvers
{
	using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

	class LinearizationSolver : public core::IFlowFieldSolver<PtrProblemTyp>
	{
	public:
		LinearizationSolver(double min_relaxation, double max_relaxtion);

		// Inherited via IFlowFieldSolver
		virtual PtrFlowField Solve(const PtrProblemTyp problem) override;

		virtual PtrFlowField Solve(const PtrProblemTyp problem, PtrFlowField initial_guess) override;

	private:
		double ComputeRelaxation(size_t relaxation_iter);

		double _start_relaxation;
		double _end_relaxation;
		double _relaxation_steps;

		std::shared_ptr<core::IColorCrossFilterProblem> _cross_filter;
		std::shared_ptr<ISunBakerLSBuilder> _linear_system_builder;
		using PtrLinearSolver = std::shared_ptr<core::ILinearSolver<float>>;
		PtrLinearSolver _linear_solver;

	};
}