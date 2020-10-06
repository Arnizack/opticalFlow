#pragma once
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"
#include"unit_tests/core_mock_adaptor/MockIArray.h"
#include"unit_tests/core_mock_adaptor/penalty/MockIPenalty.h"

namespace core
{
	namespace testing
	{
		class FakeGrayCrossPenaltyProblem : public IGrayPenaltyCrossProblem
		{
		public:
			using MockGrayImg = MockIArray<float, 2>;
			using MockColorImg = MockIArray<float, 3>;
			using MockPenaltyFunc = MockIPenalty<
				std::shared_ptr<IArray<float, 2>>>;

			FakeGrayCrossPenaltyProblem();

			MockGrayImg FirstMockImage;
			MockGrayImg SecondMockImage;
			MockColorImg CrossMockImage;
			MockPenaltyFunc PenaltyMockFunc;
		};
	}
}

