#pragma once
#include"core/solver/problem/ILinearProblem.h"

#include "unit_tests/core_mock_adaptor/MockIArray.h"
#include "unit_tests/core_mock_adaptor/linalg/MockILinearOperator.h"

namespace core
{
	namespace testing
	{
		class FakeLinearProblem : public core::ILinearProblem<double>
		{
			using MockPtrVector = MockIArray<double, 1>;
			using MockPtrLinearOperator = MockILinearOperator<MockPtrVector, MockPtrVector>;

		public:
			FakeLinearProblem();
				/*: MockVector(), MockLinearOperator(), ILinearProblem<double>()
			{
				Vector = std::make_shared<core::testing::MockIArray<double, 1>>();
				LinearOperator = std::make_shared<core::testing::MockILinearOperator<MockPtrVector, MockPtrVector>>();
			}*/

			MockPtrVector MockVector;
			MockPtrLinearOperator MockLinearOperator;
		};
	}
}