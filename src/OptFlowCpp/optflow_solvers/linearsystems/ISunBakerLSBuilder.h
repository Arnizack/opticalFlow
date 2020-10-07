#pragma once
#include"core/solver/ILinearSolver.h"


namespace optflow_solvers
{
	using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
	class ISunBakerLSBuilder
	{
	public:
		
		virtual void SetFramePair(PtrGrayImg first_image, PtrGrayImg second_image) = 0;
		virtual void UpdateParameter(PtrFlowField linearizazion_points, double relaxation) = 0;
		virtual std::shared_ptr<core::ILinearProblem<float>> Create() = 0;
	};
}