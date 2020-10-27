#pragma once
#include"FakeLinearProblem.h"

using MockPtrVector = std::shared_ptr<core::testing::MockIArray<double, 1>>;
using PtrVector = std::shared_ptr<core::IArray<double, 1>>;

core::testing::FakeLinearProblem::FakeLinearProblem()
	: MockVector(), MockLinearOperator(), ILinearProblem<double>()
{
	Vector = std::make_shared<core::testing::MockIArray<double, 1>>();
	LinearOperator = std::make_shared<core::testing::MockILinearOperator<PtrVector, PtrVector>>();
}