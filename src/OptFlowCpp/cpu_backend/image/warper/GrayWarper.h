#pragma once
#include"core/image/IGrayWarper.h"
#include<vector>
#include"../inner/DerivativeCalculator.h"

namespace cpu_backend
{
	class GrayWarper : public core::IGrayWarper
	{
		/*
		Border: The is clipped, so it can not go out side the image
		*/
		using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
		using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
	public:

		// Inherited via IGrayWarper
		virtual void SetImage(PtrGrayImg image) override;
		virtual PtrGrayImg Warp(PtrFlowField flow) override;

		
	private:
		std::shared_ptr<Array<float,2>> _image;

		
		
	};
}