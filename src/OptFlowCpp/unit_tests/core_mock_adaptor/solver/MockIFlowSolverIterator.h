#pragma once
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"
#include"core/solver/IFlowSolverIterator.h"
#include"gmock/gmock.h"
namespace core
{
	namespace testing
	{ 
		
		class MockIFlowSolverIterator : public core::IFlowSolverIterator<IGrayPenaltyCrossProblem>
		{
		public:
			MockIFlowSolverIterator(size_t range,
			std::shared_ptr<IFlowFieldSolver<std::shared_ptr<IGrayPenaltyCrossProblem>>> solver);

			
			std::shared_ptr<IFlowFieldSolver<std::shared_ptr<IGrayPenaltyCrossProblem>>> Current() override;

			virtual void Increament() override;
			virtual bool IsEnd() override;
			virtual size_t Range() override;

		private:
			size_t _range;
			size_t _index = 0;
			std::shared_ptr<IFlowFieldSolver<std::shared_ptr<IGrayPenaltyCrossProblem>>> _solver;
			// Inherited via IFlowSolverIterator
			
		};
	}
}