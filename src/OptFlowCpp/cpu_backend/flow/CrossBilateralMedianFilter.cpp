#pragma once
#include"pch.h"
#include"CrossBilateralMedianFilter.h"

namespace cpu_backend
{
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
	using PtrColorImage = std::shared_ptr<core::IArray<float, 3>>;
	PtrFlowField CrossBilateralMedianFilter::Apply(const PtrFlowField vec)
	{
		return PtrFlowField();
	}
	void CrossBilateralMedianFilter::ApplyTo(PtrFlowField dst, const PtrFlowField vec)
	{

	}
}