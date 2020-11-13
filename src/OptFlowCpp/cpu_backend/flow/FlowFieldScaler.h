#pragma once
#include"core/IScaler.h"
#include"core/IArrayFactory.h"
#include "../Array.h"
#include "cpu_backend/image/inner/DownScaleGaussianGrayScale.h"
#include "cpu_backend/image/inner/BicubicScale.h"
#include<memory>

namespace cpu_backend
{
	class FlowFieldScaler : public core::IScaler<core::IArray<double,3>>
	{
	public:
		FlowFieldScaler(std::shared_ptr<core::IArrayFactory<double, 3>> array_factory)
			: _array_factory(array_factory)
		{}

		// Inherited via IScaler
		virtual std::shared_ptr < core::IArray<double, 3>> Scale(const std::shared_ptr < core::IArray<double, 3>> input, const size_t& dst_width, const size_t& dst_height) override;
	private:
		inline void ScaleIn(double* input, const double& factor, const size_t& size);

		std::shared_ptr<core::IArrayFactory<double, 3>> _array_factory;
	};
}