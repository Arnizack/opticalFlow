#pragma once
#include"core/IArray.h"
#include"gmock/gmock.h"

namespace core
{
	namespace testing
	{
		template<class T, size_t DimCount>
		class MockIArray : public IArray<T, DimCount>
		{
		public:

			MockIArray(std::array<const size_t, DimCount> shape) : IArray(shape)
			{}
			MockIArray() : IArray({ 0 })
			{}
			MockIArray(const core::testing::MockIArray<T, DimCount>& copy) : IArray(copy.Shape)
			{}

			MOCK_METHOD(size_t, Size, (), (const, override));
			MOCK_METHOD(bool, CopyDataTo, (T* destination), (override));
		};
	}
}