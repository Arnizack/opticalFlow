#pragma once
#include"core/solver/IFlowSolverIterator.h"
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"
#include"core/pyramid/IPyramidBuilder.h"
#include"PyramidSolver.h"
#include<vector>
namespace optflow_solvers
{
	struct PyramidResolutions
	{
		double ScaleFactor = 0.5;
		size_t MinResolutionX = 64;
		size_t MinResolutionY = 64;
	};

	struct PyramidsResolutions
	{
		std::vector<PyramidResolutions> Resolutions;
	};

	class PyramidSolverIterator : public core::IFlowSolverIterator< core::IGrayPenaltyCrossProblem>
	{
	public:

		PyramidSolverIterator(std::shared_ptr < PyramidsResolutions> resolutions, 
			std::shared_ptr <PyramidSolver > pyramid_solver);

		virtual std::shared_ptr<core::IFlowFieldSolver<std::shared_ptr<core::IGrayPenaltyCrossProblem>>> Current() override;

		virtual void Increament() override;

		virtual bool IsEnd() override;

		virtual size_t Range() override;
		

	private:
		std::shared_ptr <PyramidSolver > _pyramid_solver;
		size_t _pyramid_index = 0;

		std::shared_ptr < PyramidsResolutions> _resolutions;

		std::shared_ptr<core::IPyramidBuilder<std::shared_ptr<core::IGrayPenaltyCrossProblem>>> _pyramid_builder;
		

		

	};
}