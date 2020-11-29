#pragma once
#include"core/solver/ILinearSolver.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{
		
		class MockILinearSolverD : public ILinearSolver<double>
		{
		public:
			using PtrVector = std::shared_ptr<IArray<double, 1>>;


			MOCK_METHOD(PtrVector, Solve ,
				(std::shared_ptr < ILinearProblem<double>> problem),(override));
			MOCK_METHOD2(Solve,
				PtrVector(std::shared_ptr < ILinearProblem<double>> problem,
					const PtrVector initial_guess));


		};
	}
}