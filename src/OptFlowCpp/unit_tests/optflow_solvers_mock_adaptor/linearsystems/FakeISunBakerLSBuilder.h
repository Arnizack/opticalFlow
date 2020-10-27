#pragma once
#include"optflow_solvers/linearsystems/ISunBakerLSBuilder.h"
#include"gmock/gmock.h"
#include"unit_tests/core_mock_adaptor/solver/problem/MockILinearProblem.h"

namespace optflow_solvers
{
	namespace testing
	{
		class FakeISunBakerLSBuilder : public ISunBakerLSUpdater
		{
		public:
			// Inherited via ISunBakerLSBuilder
			MOCK_METHOD2(SetFramePair,
				void(PtrGrayImg first_image, PtrGrayImg second_image));
			MOCK_METHOD2(UpdateParameter, 
				void(PtrFlowField linearizazion_points, double relaxation) );

			virtual std::shared_ptr<core::ILinearProblem<double>> Update() override {
				return std::make_shared<core::testing::MockILinearProblem<double>>();
			}
		};
	}
}