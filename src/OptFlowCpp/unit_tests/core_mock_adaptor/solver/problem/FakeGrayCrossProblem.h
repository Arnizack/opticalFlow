#pragma once
#include"core/solver/problem/IGrayCrossFilterProblem.h"
#include"unit_tests/core_mock_adaptor/MockIArray.h"

namespace core
{
	namespace testing
	{
		class FakeGrayCrossProblem : public IGrayCrossFilterProblem
		{
		public:
			using MockGrayImg = MockIArray<float, 2>;
			using MockColorImg = MockIArray<float, 3>;

			FakeGrayCrossProblem();

			MockGrayImg FirstMockImage;
			MockGrayImg SecondMockImage;
			MockColorImg CrossMockImage;

		};
	}
}