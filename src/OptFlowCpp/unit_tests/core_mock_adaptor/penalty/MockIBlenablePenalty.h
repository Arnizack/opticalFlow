#pragma once
#include"core/penalty/IBlendablePenalty.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{

		template<class T>
		class MockIBlendablePenalty : public IBlendablePenalty<T>
		{
		public:
			MOCK_METHOD(T, ValueAt, (const T& x), (override));

			MOCK_METHOD(T, FirstDerivativeAt, (const T& x), (override));

			MOCK_METHOD(T, SecondDerivativeAt, (const T& x), (override));

			MOCK_METHOD(void, SetBlendFactor, (double blend_factor), (override));
		};
	}
}