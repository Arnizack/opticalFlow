#pragma once
#include<memory>
#include"core/solver/IFlowFieldSolver.h"
#include"core/IArrayFactory.h"
#include"HornSchunckSettings.h"
#include"../solver/FlowFieldSolverBase.h"

namespace hsflow
{
	class HornSchunckSolver : public optflow::FlowFieldSolverBase<double,3,HSSettings>
	{
	public:
		HornSchunckSolver();

		// Inherited via IFlowFieldSolver
		PtrFlowField Solve(const PtrImage first_frame, 
			const PtrImage second_frame, const PtrFlowField InitialGuess,
			HSSettings settings) override final;
	
	

	};
}