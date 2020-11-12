#include"pch.h"
#include "PyramidSolverIterator.h"


namespace optflow_solvers
{
    PyramidSolverIterator::PyramidSolverIterator(std::shared_ptr < PyramidsResolutions> resolutions, 
                std::shared_ptr <PyramidSolver > pyramid_solver) 
        : _resolutions(resolutions) , _pyramid_solver(pyramid_solver)
    {
        
    }
    
    std::shared_ptr<core::IFlowFieldSolver<std::shared_ptr<core::IGrayPenaltyCrossProblem>>> PyramidSolverIterator::Current()  
    {
        PyramidResolutions resolution = _resolutions->Resolutions[_pyramid_index];
        _pyramid_builder->SetScaleFactor(resolution.ScaleFactor,
            std::array<size_t,2>{resolution.MinResolutionX,resolution.MinResolutionY});
        _pyramid_solver->SetPyramidBuilder(_pyramid_builder);
        return std::static_pointer_cast<core::IFlowFieldSolver<std::shared_ptr<core::IGrayPenaltyCrossProblem>>>(_pyramid_solver);
    }
    
    void PyramidSolverIterator::Increament() 
    {
        _pyramid_index++;
    }
    
    bool PyramidSolverIterator::IsEnd() 
    {
        return _pyramid_index == Range()-1;
    }
    
    size_t PyramidSolverIterator::Range() 
    {
        return _resolutions->Resolutions.size();
    }
}