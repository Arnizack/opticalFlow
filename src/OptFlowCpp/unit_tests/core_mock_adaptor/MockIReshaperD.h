#pragma once
#include"core/IReshaper.h"
#include"gmock/gmock.h"
namespace core
{
	namespace testing
	{
		
		class MockIReshaperD : public IReshaper< double>
		{
		public:
			MOCK_METHOD1(Reshape1D ,
				std::shared_ptr < IArray<double, 1>>(std::shared_ptr<IContainer<double>> container));

			MOCK_METHOD2(
				Reshape2D,
				std::shared_ptr < IArray<double, 2>>(std::shared_ptr<IContainer<double>> container,
					std::array<const size_t, 2> shape));

			MOCK_METHOD2(
				Reshape3D, std::shared_ptr < IArray<double, 3>>(std::shared_ptr<IContainer<double>>  container,
					std::array<const size_t, 3> shape) );
		};
	}
}