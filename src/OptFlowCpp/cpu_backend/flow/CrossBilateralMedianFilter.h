#pragma once
#include"core/flow/ICrossFlowFilter.h"
namespace cpu_backend
{
	//ToDo
	class CrossBilateralMedianFilter : public core::ICrossFlowFilter
	{
		using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
		using PtrColorImage = std::shared_ptr<core::IArray<float, 3>>;
	public:
		virtual PtrFlowField Apply(const PtrFlowField vec) override;
		virtual void ApplyTo(PtrFlowField dst, const PtrFlowField vec) override;
		virtual void SetFilterInfluence(double influence) override;
		virtual void SetCrossFilterImage(PtrColorImage image) override;

	};
}