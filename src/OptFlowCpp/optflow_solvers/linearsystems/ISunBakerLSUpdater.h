#pragma once
#include"core/solver/ILinearSolver.h"
#include"core/penalty/IPenalty.h"

namespace optflow_solvers
{
	using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
	class ISunBakerLSUpdater
	{
	public:
		
		virtual void SetFramePair(PtrGrayImg first_image, PtrGrayImg second_image) = 0;
		virtual void UpdateParameter(PtrFlowField linearization_points, double relaxation) = 0;
		virtual std::shared_ptr<core::ILinearProblem<double>> Update() = 0;
		virtual void SetPenalty(std::shared_ptr<core::IPenalty<double>> penalty) = 0;
	};
}