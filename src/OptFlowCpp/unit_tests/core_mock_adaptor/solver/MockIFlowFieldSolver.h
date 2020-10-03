#pragma once
#include"core/solver/IFlowFieldSolver.h"
#include"gmock/gmock.h"

namespace core
{
	namespace solver
	{
		namespace testing
		{
			using ProblemTyp = std::shared_ptr<problem::IGrayPenaltyCrossProblem > ;
			class MockIFlowFieldSolver : public IFlowFieldSolver<ProblemTyp>
			{
			public:
				using PtrFlowField = std::shared_ptr<IArray<double, 3> >;
				MOCK_METHOD(PtrFlowField, Solve , (const ProblemTyp problem), (override));

				MOCK_METHOD(PtrFlowField, Solve, (const ProblemTyp problem, PtrFlowField initial_guess), (override));

			};
		}
	}
}