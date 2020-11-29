#pragma once
#include"core/penalty/IPenalty.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{

		template<class T>
		class MockIPenalty : public IPenalty<T>
		{
		public:
			MOCK_METHOD(T, ValueAt, (const T& x), (override));

			MOCK_METHOD(T, FirstDerivativeAt, (const T& x), (override));

			MOCK_METHOD(T, SecondDerivativeAt, (const T& x), (override));
		};
	}
}