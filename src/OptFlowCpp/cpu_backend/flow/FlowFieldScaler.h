#pragma once
#include"core/IScaler.h"
#include"core/IArray.h"
#include<memory>

namespace cpu_backend
{
	class FlowFieldScaler : core::IScaler<core::IArray<double,3>>
	{
	public:


		// Inherited via IScaler
		virtual std::shared_ptr < core::IArray<double, 3>> Scale(const std::shared_ptr < core::IArray<double, 3>> input, const size_t& dst_width, const size_t& dst_height) override;

	};
}