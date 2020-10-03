#pragma once
#include"../framework.h"
#include"core/solver/IFlowFieldSolver.h"
#include"core/penalty/IBlendablePenalty.h"
#include"core/IArrayFactory.h"
#include"core/solver/problem/IProblemFactory.h"

namespace opticalflow_solvers
{

	namespace cs = core::solver;
	namespace cp = core::penalty;

	using ProblemTyp = std::shared_ptr<cs::problem::IGrayCrossFilterProblem>;

	class GNCPenaltySolver : public cs::IFlowFieldSolver<ProblemTyp>
	{
	public:

		
		using PtrFlowField = std::shared_ptr < core::IArray<double, 3> > ;
		using PtrStandardFlowSolver = std::shared_ptr<cs::IFlowFieldSolver<std::shared_ptr<cs::problem::IGrayPenaltyCrossProblem>>>;
		using PtrGrayScale = std::shared_ptr <core::IArray<float, 2>>;
		using PtrBlendPenalty = std::shared_ptr<cp::IBlendablePenalty<PtrGrayScale>>;
		using PtrFlowFactory = std::shared_ptr<core::IArrayFactory<double, 3>>;
		using PtrProblemFactory = std::shared_ptr<cs::problem::IProblemFactory>;

		GNCPenaltySolver(int gnc_steps, std::vector<PtrStandardFlowSolver> inner_solvers, 
			PtrBlendPenalty penalty_func,
			PtrFlowFactory flow_factory, PtrProblemFactory problem_factory);

		// Inherited via IFlowFieldSolver
		virtual PtrFlowField Solve(const ProblemTyp problem) override;

		virtual PtrFlowField Solve(const ProblemTyp problem, PtrFlowField initial_guess) override;


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