#include "FakeGrayCrossProblem.h"
#include"../../MockIArray.h"

using PtrGrayImg = std::shared_ptr < core::IArray<float, 2>>;

core::solver::problem::testing::FakeGrayCrossProblem::FakeGrayCrossProblem()
	:FirstMockImage(), SecondMockImage(), CrossMockImage(), IGrayCrossFilterProblem()
{
	FirstFrame = std::make_shared< core::testing::MockIArray<float,2>>();
	SecondFrame = std::make_shared < core::testing::MockIArray<float, 2>>();
	CrossFilterImage = std::make_shared < core::testing::MockIArray<float, 3>>();
}
