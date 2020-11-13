#include"pch.h"
#include "PyramidSolver.h"
namespace optflow_solvers
{

	using PtrProblemTyp = std::shared_ptr<core::IGrayPenaltyCrossProblem>;
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3> >;
	using PtrProblemPyramid = std::shared_ptr<core::IPyramid<PtrProblemTyp>>;

	PyramidSolver::PyramidSolver(
		std::shared_ptr<core::IArrayFactory<double, 3>> flow_factory, 
		std::shared_ptr<core::IPyramidBuilder<PtrProblemTyp>> pyramid_builder, 
		std::shared_ptr<core::IScaler<double, 3>> flow_scaler,
		std::shared_ptr<core::IFlowFieldSolver<PtrProblemTyp>> inner_solver)
		: _flow_factory(flow_factory), _pyramid_builder(pyramid_builder), 
		_flow_scaler(flow_scaler), _inner_solver(inner_solver)
	{
	}

	PtrFlowField PyramidSolver::Solve(const PtrProblemTyp problem)
	{
		size_t width = problem->FirstFrame->Shape[1];
		size_t height = problem->FirstFrame->Shape[0];
		PtrFlowField initial_guess = _flow_factory->Zeros({ 2,height,width });

		return Solve(problem, initial_guess);
	}
	PtrFlowField PyramidSolver::Solve(
		const PtrProblemTyp problem, PtrFlowField initial_guess)
	{
		auto pyramid = _pyramid_builder->Create(problem);
		return Solve(pyramid, initial_guess);
	}
	PtrFlowField PyramidSolver::Solve(PtrProblemPyramid pyramid, PtrFlowField initial_guess)
	{
		PtrFlowField initial_guess_scaled = initial_guess;
		while (!pyramid->IsEndLevel())
		{
			auto problem = pyramid->NextLevel();
			size_t width = problem->FirstFrame->Shape[1];
			size_t heigth = problem->FirstFrame->Shape[0];
			
			/*PtrFlowField*/ initial_guess_scaled = _flow_scaler->Scale(initial_guess_scaled,width,heigth);

			initial_guess_scaled = _inner_solver->Solve(problem, initial_guess_scaled);
		}
		return initial_guess_scaled;
	}
	void PyramidSolver::SetPyramidBuilder(std::shared_ptr<core::IPyramidBuilder<PtrProblemTyp>> pyramid_builder)
	{
		_pyramid_builder = pyramid_builder;
	}
}
