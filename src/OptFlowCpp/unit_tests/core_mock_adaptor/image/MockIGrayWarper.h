#pragma once
#include"core/image/IGrayWarper.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{
		class MockIGrayWarper : public IGrayWarper
		{
		public:
			using PtrGrayImg = std::shared_ptr<IArray<float, 2>>;
			using PtrFlowField = std::shared_ptr<IArray<double, 3>>;
			MOCK_METHOD2(Warp, PtrGrayImg(PtrGrayImg Image, PtrFlowField Flow));

		};
	}
}