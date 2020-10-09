#pragma once
#include"core/image/IGrayWarper.h"

namespace cpu_backend
{
	class GrayWarper : public core::IGrayWarper
	{
		using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
		using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
	public:

		// Inherited via IGrayWarper
		virtual void SetImage(PtrGrayImg Image) override;
		virtual PtrGrayImg Warp(PtrFlowField Flow) override;

	private:
		PtrGrayImg _image;

	};
}