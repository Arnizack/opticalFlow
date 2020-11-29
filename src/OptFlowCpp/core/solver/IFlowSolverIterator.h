#pragma once
#include"IFlowFieldSolver.h"

namespace core
{
	template<class _TProblem>
	class IFlowSolverIterator
	{
	public:
		virtual std::shared_ptr<IFlowFieldSolver<std::shared_ptr<_TProblem>>> Current() = 0;
		virtual void Increament() = 0;
		virtual bool IsEnd() = 0;
		virtual size_t Range() = 0;

	};
}