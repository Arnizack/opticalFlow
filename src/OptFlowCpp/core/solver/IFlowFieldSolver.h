#pragma once
#include"problem/IColor2FrameProblem.h"
#include"problem/IGray2FrameProblem.h"
#include"problem/IGrayCrossFilterProblem.h"
#include"problem/IGrayPenaltyCrossProblem.h"
#include"problem/IGrayPenaltyProblem.h"

namespace core
{
	namespace solver
	{
		template<class ProblemTyp>
		class IFlowFieldSolver
		{
		public:
			using PtrFlowField = std::shared_ptr<IArray<double, 3> >;
			
			virtual PtrFlowField Solve(const ProblemTyp problem) = 0;

			virtual PtrFlowField Solve(const ProblemTyp problem, PtrFlowField initial_guess) = 0;
		};
	}
}