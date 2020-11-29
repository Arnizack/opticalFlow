#include"MockIFlowSolverIterator.h"
namespace core
{
    namespace testing
    {
        MockIFlowSolverIterator::MockIFlowSolverIterator(size_t range, std::shared_ptr<IFlowFieldSolver<std::shared_ptr<IGrayPenaltyCrossProblem>>> solver) 
           : _range(range), _solver(solver)
        {
            
        }
        
        std::shared_ptr<IFlowFieldSolver<std::shared_ptr<IGrayPenaltyCrossProblem>>> MockIFlowSolverIterator::Current() 
        {
            return _solver;
        }
    
        void MockIFlowSolverIterator::Increament() 
        {
            _index++;
        }
        
        bool MockIFlowSolverIterator::IsEnd() 
        {
            return _range == _index -1;
        }
        size_t MockIFlowSolverIterator::Range()
        {
            return _range;
        }
    }
}