#pragma once
#include"core/solver/problem/IProblemFactory.h"
#include"gmock/gmock.h"
namespace core
{
	namespace solver
	{
		namespace problem
		{
			namespace testing
			{ 
				namespace csp = core::solver::problem;
				class MockIProblemFactory : public IProblemFactory
				{
				public:
					MOCK_METHOD( std::shared_ptr<csp::IGrayPenaltyCrossProblem>, 
						CreateGrayPenaltyCrossProblem,(), (override)) ;
				};
			}
		}
	}
}