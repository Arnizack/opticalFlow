#pragma once
#include "..\IArray.h"
#include "..\linalg\ILinearOperator.h"
#include <memory>
#include"problem/ILinearProblem.h"

namespace core
{

	template<class InnerTyp>
	class ILinearSolver
	{
	public:
		using PtrVector = std::shared_ptr<IArray<InnerTyp, 1>>;
		

		virtual PtrVector Solve(std::shared_ptr < ILinearProblem<InnerTyp>> problem) = 0;
		virtual PtrVector Solve(std::shared_ptr < ILinearProblem<InnerTyp>> problem, const PtrVector initial_guess) = 0;


	};

}