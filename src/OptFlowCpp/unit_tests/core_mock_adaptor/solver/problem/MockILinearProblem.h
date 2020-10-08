#pragma once
#include"core/solver/problem/ILinearProblem.h"

namespace core
{
	template<class InnerTyp>
	class MockILinearProblem : public ILinearProblem<InnerTyp>
	{
		using PtrVector = std::shared_ptr<IArray<InnerTyp, 1>>;
		using PtrLinearOperator = std::shared_ptr<ILinearOperator<PtrVector, PtrVector>>;

		PtrVector Vector{nullptr};
		PtrLinearOperator LinearOperator{nullptr};
	};
}