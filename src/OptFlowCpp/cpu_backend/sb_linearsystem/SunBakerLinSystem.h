#pragma once
#include"core/linalg/ILinearOperator.h"
#include"core/IArray.h"
#include<memory>

namespace cpu_backend
{
	using PtrGrayImg = std::shared_ptr<core::IArray<double, 1>>;
	class SunBakerLinSystem : core::ILinearOperator<PtrGrayImg,PtrGrayImg>
	{
	public:

		// Inherited via ILinearOperator
		virtual PtrGrayImg Apply(const PtrGrayImg vec) override;
		virtual void ApplyTo(PtrGrayImg dst, const PtrGrayImg vec) override;
		virtual ILinearOperator<PtrGrayImg, PtrGrayImg> Transpose() override;
		virtual bool IsSymetric() override;
	private:
		size_t width;
		size_t height;
	};
}