#pragma once
#include"cpu_backend/Base.h"
#include"CrossBilateralMedianFilter.h"
#include"../Array.h"
#include"inner/LogOcclusion.h"
#include"inner/BilateralMedian.h"
#include"../image/inner/WarpLinearColorImage.h"
#include"inner/BilateralMedian.h"


namespace cpu_backend
{
	using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
	using PtrColorImage = std::shared_ptr<core::IArray<float, 3>>;
	CrossBilateralMedianFilter::CrossBilateralMedianFilter(
		std::shared_ptr<DerivativeCalculator<double>> flow_deriv_calc,
		std::shared_ptr<CrossMedianFilterSettings> settings)
		: _flow_deriv_calc(flow_deriv_calc),
		_sigma_div(settings->SigmaDiv),
		_sigma_error(settings->SigmaError),
		_filter_influence(settings->FilterInfluence),
		_sigma_distance(settings->SigmaDistance),
		_sigma_color(settings->SigmaColor),
		_filter_length(settings->FilterLength)

	{

	}
	PtrFlowField CrossBilateralMedianFilter::Apply(const PtrFlowField vec)
	{
		return PtrFlowField();
	}
	void CrossBilateralMedianFilter::ApplyTo(PtrFlowField dst, const PtrFlowField vec)
	{
		OPF_LOG_TRACE("CrossBilateralMedianFilter");

		auto destination = std::static_pointer_cast<Array<double, 3>>(dst);
		auto flow = std::static_pointer_cast<Array<double, 3>>(vec);
		
		

		WarpLinearColorImage(_image_warped->Data(), _image->Data(), flow->Data(), _width, _height, _color_channel_count);
		double* ptr_flow = flow->Data();
		double* ptr_flow_div = _flow_divergence->Data();
		_flow_deriv_calc->FlowDivergence(ptr_flow, _width, _height, ptr_flow_div);

		ComputeLogOcclusion(_log_occlusion_map->Data(), _image->Data(), _image_warped->Data(), _flow_divergence->Data(),
			_width, _height, _color_channel_count, _sigma_div, _sigma_error);


		BilateralMedian(flow->Data(),flow->Data(),_log_occlusion_map->Data(),_image->Data(),_filter_influence, _auxiliary_influence,
			_sigma_distance, _sigma_color, _filter_length, _width, _height, _color_channel_count, destination->Data());
		
	}
	void CrossBilateralMedianFilter::SetAuxiliaryInfluence(double influence)
	{
		_auxiliary_influence = influence;
	}
	void CrossBilateralMedianFilter::SetCrossFilterImage(PtrColorImage image)
	{
		_image = std::static_pointer_cast<Array<float, 3>>(image);
		_image_warped = std::make_shared<Array<float, 3>>(image->Shape);
		size_t width = image->Shape[2];
		size_t height = image->Shape[1];
		_color_channel_count = image->Shape[0];
		SetUpCache(width, height);
	}
	void CrossBilateralMedianFilter::SetUpCache(size_t width, size_t height)
	{
		if(_width != width || _height != height)
		{
			_width = width;
			_height = height;
			std::array<const size_t, 2> shape = { height,width };
			_log_occlusion_map = std::make_shared<Array<double, 2>>(shape);
			_flow_divergence = std::make_shared<Array<double, 2>>(shape);
			
		}
	}
}