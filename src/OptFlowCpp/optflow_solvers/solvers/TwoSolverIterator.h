#pragma once
#include"core/solver/IFlowSolverIterator.h"

namespace optflow_solvers
{
	template<class T>
	class TwoSolverIterator : public core::IFlowSolverIterator< T>
	{
	private:
		std::shared_ptr<core::IFlowFieldSolver<std::shared_ptr<T>>> _first_solver;
		std::shared_ptr<core::IFlowFieldSolver<std::shared_ptr<T>>> _second_solver;
		bool _is_second = false;
	public:
		TwoSolverIterator(std::shared_ptr<core::IFlowFieldSolver<std::shared_ptr<T>>> first_solver,
			std::shared_ptr<core::IFlowFieldSolver<std::shared_ptr<T>>> second_solver)
			: _first_solver(first_solver),_second_solver(second_solver)
		{

		}
		// Inherited via IFlowSolverIterator
		virtual std::shared_ptr<core::IFlowFieldSolver<std::shared_ptr<T>>> Current() override
		{
			if (_is_second)
				return _second_solver;
			return _first_solver;
		}
		virtual void Increament() override
		{
			_is_second = true;
		}
		virtual bool IsEnd() override
		{
			return _is_second;
		}
		virtual size_t Range() override
		{
			return 2;
		}
	};
}