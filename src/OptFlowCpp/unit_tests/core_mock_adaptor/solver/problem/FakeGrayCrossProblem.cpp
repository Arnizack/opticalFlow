#include "FakeGrayCrossProblem.h"

using PtrGrayImg = std::shared_ptr < core::IArray<float, 2>>;

core::solver::problem::testing::FakeGrayCrossProblem::FakeGrayCrossProblem()
	:FirstMockImage(), SecondMockImage(), CrossMockImage(), IGrayCrossFilterProblem()
{
	FirstFrame = std::shared_ptr < core::IArray<float, 2>>(&FirstMockImage);
	SecondFrame = std::shared_ptr < core::IArray<float, 2>>(&SecondMockImage);
	CrossFilterImage = std::shared_ptr < core::IArray<float, 3>>(&CrossMockImage);
}
