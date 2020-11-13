#pragma once
#include"core/solver/IFlowFieldSolver.h"
#include"core/solver/problem/IGrayCrossFilterProblem.h"
#include"core/flow/ICrossFlowFilter.h"
#include"core/solver/ILinearSolver.h"
#include"../linearsystems/ISunBakerLSUpdater.h"
#include"core/IReshaper.h"
#include"core/image/IGrayWarper.h"
#include"core/IArrayFactory.h"
#include"core/linalg/IArithmeticBasic.h"
#include"core/flow/ICrossFlowFilter.h"

namespace optflow_solvers
{
	using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

	struct LinearizationSolverSettings
	{
		double StartRelaxation = 1e-04 / 255.0;
		double EndRelaxation = 1e-01 / 255.0;
		double RelaxationSteps = 3;
	};

	class LinearizationSolver : public core::IFlowFieldSolver<PtrProblemTyp>
	{
		using PtrLinearSolver = std::shared_ptr<core::ILinearSolver<double>>;
	public:
		LinearizationSolver(
			std::shared_ptr<LinearizationSolverSettings> settings,
			std::shared_ptr<core::ICrossFlowFilter> cross_filter,
			std::shared_ptr<ISunBakerLSUpdater> linear_system_builder,
			PtrLinearSolver linear_solver,
			std::shared_ptr<core::IReshaper<double>> flow_reshaper,
			std::shared_ptr<core::IGrayWarper> warper,
			std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory,
			std::shared_ptr<core::IArithmeticBasic<double, 3>> flow_arithmetic
		);

		// Inherited via IFlowFieldSolver
		virtual PtrFlowField Solve(const PtrProblemTyp problem) override;

		virtual PtrFlowField Solve(const PtrProblemTyp problem, PtrFlowField initial_guess) override;

	private:
		double ComputeRelaxation(size_t relaxation_iter);

		double _start_relaxation;
		double _end_relaxation;
		double _relaxation_steps;

		std::shared_ptr<core::ICrossFlowFilter> _cross_filter;
		std::shared_ptr<ISunBakerLSUpdater> _linear_system_updater;
		PtrLinearSolver _linear_solver;
		std::shared_ptr<core::IReshaper<double>> _flow_reshaper;
		std::shared_ptr<core::IGrayWarper> _warper;
		std::shared_ptr<core::IArrayFactory<double, 3>> _flow_factory;
		std::shared_ptr<core::IArithmeticBasic<double, 3>> _flow_arithmetic;
	};
}