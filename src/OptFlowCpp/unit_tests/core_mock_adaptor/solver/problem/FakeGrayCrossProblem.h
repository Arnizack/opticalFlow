#pragma once
#include"core/solver/problem/IGrayCrossFilterProblem.h"
#include"unit_tests/core_mock_adaptor/MockIArray.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			namespace testing
			{
				class FakeGrayCrossProblem : public IGrayCrossFilterProblem
				{
				public:
					using MockGrayImg = core::testing::MockIArray<float, 2>;
					using MockColorImg = core::testing::MockIArray<float, 3>;

					FakeGrayCrossProblem();

					MockGrayImg FirstMockImage;
					MockGrayImg SecondMockImage;
					MockColorImg CrossMockImage;

					

				};
			}
		}
	}
}