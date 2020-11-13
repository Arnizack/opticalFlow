#pragma once
#include"core/IArray.h"
#include"core/IArrayFactory.h"
#include"core/solver/IFlowFieldSolver.h"
#include"core/pyramid/IPyramidBuilder.h"
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"
#include"core/IScaler.h"
#include"core/solver/IFlowFieldSolver.h"

namespace optflow_solvers
{
	
	using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;
	using PtrProblemPyramid = std::shared_ptr<core::IPyramid<core::IGrayPenaltyCrossProblem>>;

	class PyramidSolver : public core::IFlowFieldSolver<PtrProblemTyp>
	{
	public:

		PyramidSolver(
			std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory,
			std::shared_ptr<core::IPyramidBuilder< core::IGrayPenaltyCrossProblem>> pyramid_builder,
			std::shared_ptr<core::IScaler<core::IArray<double, 3>>> flow_scaler,
			std::shared_ptr<core::IFlowFieldSolver<PtrProblemTyp>> inner_solver
		);

		

		virtual PtrFlowField Solve(const PtrProblemTyp problem) final override;

		virtual PtrFlowField Solve(
			const PtrProblemTyp problem, PtrFlowField initial_guess) final override;
	
		PtrFlowField Solve(PtrProblemPyramid pyramid,
			PtrFlowField initial_guess );

		void SetPyramidBuilder(std::shared_ptr<core::IPyramidBuilder< core::IGrayPenaltyCrossProblem>> pyramid_builder);


	private:
		std::shared_ptr<core::IArrayFactory<double, 3>> _flow_factory;
		std::shared_ptr<core::IPyramidBuilder< core::IGrayPenaltyCrossProblem>> _pyramid_builder;
		std::shared_ptr<core::IScaler<core::IArray<double, 3>>> _flow_scaler;
		std::shared_ptr<core::IFlowFieldSolver<PtrProblemTyp>> _inner_solver;
	};

}