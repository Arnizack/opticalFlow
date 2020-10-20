#pragma once
#include"optflow_solvers/linearsystems/ISunBakerLSBuilder.h"
namespace cpu_backend
{
	class SunBakerLSUpdater : optflow_solvers::ISunBakerLSUpdater
	{
	public:
		using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
		using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
		// Inherited via ISunBakerLSBuilder
		virtual void SetFramePair(PtrGrayImg first_image, PtrGrayImg second_image) override;
		virtual void UpdateParameter(PtrFlowField linearization_points, double relaxation) override;
		virtual std::shared_ptr<core::ILinearProblem<double>> Update() override;
		
	};
}