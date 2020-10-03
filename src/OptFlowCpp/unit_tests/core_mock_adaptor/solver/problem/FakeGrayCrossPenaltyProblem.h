#pragma once
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"
#include"unit_tests/core_mock_adaptor/MockIArray.h"
#include"unit_tests/core_mock_adaptor/penalty/MockIPenalty.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			namespace testing
			{
				class FakeGrayCrossPenaltyProblem : public IGrayPenaltyCrossProblem
				{
				public:
					using MockGrayImg = core::testing::MockIArray<float, 2>;
					using MockColorImg = core::testing::MockIArray<float, 3>;
					using MockPenaltyFunc = core::penalty::testing::MockIPenalty<
						std::shared_ptr<core::IArray<float,2>>>;

					FakeGrayCrossPenaltyProblem();

					MockGrayImg FirstMockImage;
					MockGrayImg SecondMockImage;
					MockColorImg CrossMockImage;
					MockPenaltyFunc PenaltyMockFunc;
				};
			}
		}
	}
}

