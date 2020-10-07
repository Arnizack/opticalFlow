#pragma once
#include"../../IArray.h"
#include"../../linalg/ILinearOperator.h"
#include<memory>

namespace core
{
	template<class InnerTyp>
	class ILinearProblem
	{
	public:
		using PtrVector = std::shared_ptr<IArray<InnerTyp, 1>>;
		using PtrLinearOperator = std::shared_ptr<ILinearOperator<PtrVector, PtrVector>>;

		PtrVector Vector;
		PtrLinearOperator LinearOperator;

	};
}