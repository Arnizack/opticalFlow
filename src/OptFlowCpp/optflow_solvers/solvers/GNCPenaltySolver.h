#pragma once
#include"../framework.h"
#include"core/solver/IFlowFieldSolver.h"
#include"core/penalty/IBlendablePenalty.h"
#include"core/IArrayFactory.h"
#include"core/solver/problem/IProblemFactory.h"

namespace optflow_solvers
{

	

	using ProblemTyp = std::shared_ptr<core::IGrayCrossFilterProblem>;

	class GNCPenaltySolver : public core::IFlowFieldSolver<ProblemTyp>
	{

		using PtrFlowField = std::shared_ptr < core::IArray<double, 3> > ;

		using PtrStandardFlowSolver = std::shared_ptr<
			core::IFlowFieldSolver<std::shared_ptr<
			core::IGrayPenaltyCrossProblem>>>;
		
		using PtrGrayScale = std::shared_ptr <core::IArray<float, 2>>;

		using PtrBlendPenalty = std::shared_ptr<
			core::IBlendablePenalty<double>>;

		using PtrFlowFactory = std::shared_ptr<core::IArrayFactory<double, 3>>;

		using PtrProblemFactory = std::shared_ptr<core::IProblemFactory>;

	public:
	
		GNCPenaltySolver(int gnc_steps, std::vector<PtrStandardFlowSolver> inner_solvers, 
			PtrBlendPenalty penalty_func,
			PtrFlowFactory flow_factory, PtrProblemFactory problem_factory);

		virtual PtrFlowField Solve(const ProblemTyp problem) override;

		virtual PtrFlowField Solve(
			const ProblemTyp problem, PtrFlowField initial_guess) override;


	private:

		double ComputeBlendFactor(int gnc_iter, int gnc_steps);

		PtrStandardFlowSolver GetFlowSolverAt(int gnc_iter);

		int _gnc_steps;
		std::vector<PtrStandardFlowSolver> _inner_solvers;
		PtrBlendPenalty _penalty_func;
		PtrFlowFactory _flow_factory;
		PtrProblemFactory _problem_factory;
	};

}