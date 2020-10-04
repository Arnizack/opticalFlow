#include "FakeGrayCrossPenaltyProblem.h"
#include"../../penalty/MockIPenalty.h"
using PtrGrayImg = std::shared_ptr < core::IArray<float, 2>>;

core::solver::problem::testing::FakeGrayCrossPenaltyProblem::FakeGrayCrossPenaltyProblem()
	:FirstMockImage(), SecondMockImage(), CrossMockImage(), PenaltyMockFunc() , IGrayPenaltyCrossProblem()
{
	FirstFrame = std::make_shared< core::testing::MockIArray<float, 2>>();
	SecondFrame = std::make_shared < core::testing::MockIArray<float, 2>>();
	CrossFilterImage = std::make_shared < core::testing::MockIArray<float, 3>>();
	PenaltyFunc = std::make_shared < core::penalty::testing::MockIPenalty<
		std::shared_ptr<core::IArray<float, 2>>>>();

}