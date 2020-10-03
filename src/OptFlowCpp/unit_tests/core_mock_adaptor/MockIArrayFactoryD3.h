#pragma once

#include"core/IArrayFactory.h"
#include"core/IArray.h"
#include"gmock/gmock.h"
namespace core
{
	namespace testing
	{

		class MockIArrayFactoryD3 : public IArrayFactory<double,3>
		{
		public:
			using PtrArray = std::shared_ptr < IArray<double, 3>>;
			
			MOCK_METHOD1(Zeros, std::shared_ptr < IArray<double, 3>>(std::array<const size_t, 3> shape));
			
			
			MOCK_METHOD2(
				Full ,
				std::shared_ptr < IArray<double, 3>>(const double& fill_value, std::array<const size_t, 3> shape)
			);


			


		};
	}

}