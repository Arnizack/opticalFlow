#pragma once
#include"core/solver/problem/IProblemFactory.h"
#include"gmock/gmock.h"
namespace core
{
	namespace testing
	{
		class MockIProblemFactory : public IProblemFactory
		{
		public:
			MOCK_METHOD(std::shared_ptr<IGrayPenaltyCrossProblem>,
				CreateGrayPenaltyCrossProblem, (), (override));

			

			MOCK_METHOD2(
				CreateGrayCrossFilterProblem, std::shared_ptr<IGrayCrossFilterProblem>(
					std::shared_ptr<IArray<float, 3>> first_image, std::shared_ptr<IArray<float, 3>> sconde_image
					)
			);
		};
	}
}