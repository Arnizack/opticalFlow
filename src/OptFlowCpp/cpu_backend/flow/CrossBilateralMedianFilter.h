#pragma once
#include"core/flow/ICrossFlowFilter.h"
#include"core/image/IGrayWarper.h"
#include"../image/inner/DerivativeCalculator.h"
#include"../Array.h"
namespace cpu_backend
{
	struct CrossMedianFilterSettings
	{
		double SigmaDiv = 0.3;
		double SigmaError = 20;
		double FilterInfluence = 5;
		double SigmaDistance = 7;
		double SigmaColor = 7.0 /255;
		int FilterLength = 15;
	};
	//ToDo
	class CrossBilateralMedianFilter : public core::ICrossFlowFilter
	{
		using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;
		using PtrColorImage = std::shared_ptr<core::IArray<float, 3>>;
	public:
		CrossBilateralMedianFilter(std::shared_ptr<DerivativeCalculator<double>> flow_deriv_calc,
			std::shared_ptr<CrossMedianFilterSettings> settings
			);

		virtual PtrFlowField Apply(const PtrFlowField vec) override;
		virtual void ApplyTo(PtrFlowField dst, const PtrFlowField vec) override;
		virtual void SetAuxiliaryInfluence(double influence) override;
		virtual void SetCrossFilterImage(PtrColorImage image) override;
		

	private:
		void SetUpCache(size_t width, size_t height);
		
		std::shared_ptr<Array<float, 3>> _image = nullptr;
		std::shared_ptr<Array<float, 3>> _image_warped = nullptr;
		std::shared_ptr<Array<double, 2>> _log_occlusion_map = nullptr;
		std::shared_ptr<Array<double, 2>> _flow_divergence = nullptr;


		std::shared_ptr<DerivativeCalculator<double>> _flow_deriv_calc;

		size_t _width = 0;
		size_t _height = 0;
		size_t _color_channel_count = 0;
		double _auxiliary_influence;
		double _sigma_div;
		double _sigma_error;
		double _filter_influence= 0;
		double _sigma_distance;
		double _sigma_color;
		int _filter_length = 15;

	};
}