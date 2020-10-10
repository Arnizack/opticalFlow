#pragma once
#include"core/image/IGrayWarper.h"
#include<vector>
#include"../inner/DerivativeCalculator.h"

namespace cpu_backend
{
	class GrayWarper : public core::IGrayWarper
	{
		using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
		using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
	public:

		// Inherited via IGrayWarper
		virtual void SetImage(PtrGrayImg image) override;
		virtual PtrGrayImg Warp(PtrFlowField flow) override;

		std::unique_ptr<std::vector<std::array<float, 16>>> CreateBicubicLookup(
			std::shared_ptr<Array<float, 2>> image);
	private:
		std::shared_ptr<DerivativeCalculator> _derivative_calculator;
		std::unique_ptr<std::vector<std::array<float, 16>>> _lookup;
		
	};
}