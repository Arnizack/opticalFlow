#pragma once
#include "pch.h"
#include "FlowFieldScaler.h"

namespace cpu_backend
{
	std::shared_ptr<core::IArray<double, 3>> FlowFieldScaler::Scale(const std::shared_ptr<core::IArray<double, 3>> input, const size_t& dst_width, const size_t& dst_height)
	{
		auto in_width = input->Shape[2];
		auto in_height = input->Shape[1];

		auto temp = std::dynamic_pointer_cast<Array<double, 3>>(input);
		auto in = std::make_shared<Array<double, 3>>(*input);
		auto output = std::dynamic_pointer_cast<Array<double, 3>>(_array_factory->Zeros({ 2, dst_height, dst_width}));

		const size_t in_wh = in_width * in_height;

		// X
		double factor = (double)dst_width / (double)in_width;
		ScaleIn(in->Data(), factor, in_wh);
		// Y
		factor = (double)dst_height / (double)in_height;
		ScaleIn(in->Data() + in_wh, factor, in_wh);

		if (dst_width < in_width && dst_height < in_height)
		{
			_inner::DownScaleGaussianGrayScale<double>(in->Data(), in_width, in_height, dst_width, dst_height, output->Data());
			_inner::DownScaleGaussianGrayScale<double>(in->Data() + in_wh, in_width, in_height, dst_width, dst_height, output->Data() + (dst_width * dst_height));
		}
		else
		{
			_inner::BicubicGrayScale<double>(in->Data(), output->Data(), in_width, in_height, dst_width, dst_height);
			_inner::BicubicGrayScale<double>(in->Data() + in_wh, output->Data() + (dst_width * dst_height), in_width, in_height, dst_width, dst_height);
		}

		return output;
	}

	inline void FlowFieldScaler::ScaleIn(double* input, const double& factor, const size_t& size)
	{
		for (size_t i = 0; i < size; i++)
		{
			input[i] *= factor;
		}
	}
}