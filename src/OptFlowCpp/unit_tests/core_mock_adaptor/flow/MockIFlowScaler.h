#pragma once
#include"core/IScaler.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{
		class MockIFlowScaler : public IScaler<double,3>
		{
		public:

			using PtrArray = std::shared_ptr<IArray<double, 3>>;

			MOCK_METHOD(PtrArray, Scale, (const PtrArray input, const size_t& dst_width,
				const size_t& dst_height), (override));

		};
	}
}