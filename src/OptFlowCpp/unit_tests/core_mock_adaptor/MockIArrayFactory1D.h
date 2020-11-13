#pragma once
#include"core/IArrayFactory.h"
#include"core/IArray.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{

		class MockIArrayFactory1D : public IArrayFactory<double, 1>
		{
			using PtrArray = std::shared_ptr < IArray<double, 1>>;

		public:
			MOCK_METHOD1(Zeros, PtrArray(std::array<const size_t, 1> shape));


			MOCK_METHOD2(Full, PtrArray(const double& fill_value, std::array<const size_t, 1> shape));
			MOCK_METHOD2(CreateFromSource, PtrArray(const double* source, std::array<const size_t, 1> shape));
		};
	}
}