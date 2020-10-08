#pragma once
#include"gmock/gmock.h"
#include"core/flow/ICrossFlowFilter.h"

namespace core
{
	namespace testing
	{
		class MockICrossFlowFilter : public ICrossFlowFilter
		{
		public:
			using PtrFlowField = std::shared_ptr<IArray<double, 3>>;
			MOCK_METHOD(void, SetFilterInfluence, (double influence), (override));
			MOCK_METHOD(void, SetCrossFilterImage, (PtrColorImage image), (override));
			MOCK_METHOD(PtrFlowField, Apply ,(const PtrFlowField vec),(override));
			MOCK_METHOD( void, ApplyTo ,(PtrFlowField dst, const PtrFlowField vec),(override));
		};
	}
}