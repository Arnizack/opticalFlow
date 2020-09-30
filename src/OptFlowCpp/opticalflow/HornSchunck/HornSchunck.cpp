#pragma once
#include"pch.h"
#include "HornSchunck.h"


using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
using PtrImage = std::shared_ptr<core::IArray<double, 3>>;

hsflow::HornSchunckSolver::HornSchunckSolver()
    : FlowFieldSolverBase(nullptr)
{

}



PtrFlowField hsflow::HornSchunckSolver::Solve(const PtrImage first_frame, 
    const PtrImage second_frame, const PtrFlowField InitialGuess,
    HSSettings settings)
{
    return PtrFlowField();
}
