#pragma once
#include "core/linalg/ILinearOperator.h"

#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{
		template<class InputTyp, class OutputTyp>
		class MockILinearOperator : public core::ILinearOperator<InputTyp, OutputTyp>
		{
		public:
			MOCK_METHOD(OutputTyp, Apply, (const InputTyp vec), (override));

			MOCK_METHOD(void, ApplyTo, (OutputTyp dst, const InputTyp vec), (override));

			MOCK_METHOD((std::shared_ptr<core::ILinearOperator<OutputTyp, InputTyp>>), Transpose, (), (override));

			MOCK_METHOD(bool, IsSymetric, (), (override));
		};
	}
}