#include "FakeGrayCrossPenaltyProblem.h"
using PtrGrayImg = std::shared_ptr < core::IArray<float, 2>>;

core::solver::problem::testing::FakeGrayCrossPenaltyProblem::FakeGrayCrossPenaltyProblem()
	:FirstMockImage(), SecondMockImage(), CrossMockImage(), PenaltyMockFunc() , IGrayPenaltyCrossProblem()
{
	FirstFrame = std::shared_ptr < core::IArray<float, 2>>(&FirstMockImage);
	SecondFrame = std::shared_ptr < core::IArray<float, 2>>(&SecondMockImage);
	CrossFilterImage = std::shared_ptr < core::IArray<float, 3>>(&CrossMockImage);
	PenaltyFunc = std::shared_ptr<core::penalty::IPenalty< PtrGrayImg>>(&PenaltyMockFunc);
}