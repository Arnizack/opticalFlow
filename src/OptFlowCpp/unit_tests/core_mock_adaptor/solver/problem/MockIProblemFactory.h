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
		};
	}
}