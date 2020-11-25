#pragma once
#include "core/IScaler.h"
#include "core/solver/problem/IGrayPenaltyCrossProblem.h"
#include "core/solver/problem/IProblemFactory.h"
#include "core/IArrayFactory.h"
#include "../image/inner/DownScaleGaussianGrayScale.h"
#include "../image/inner/BicubicScale.h"
#include "../Array.h"
#include"utilities/debug_helper/ImageLogger.h"

namespace cpu_backend
{
	class GrayPenaltyCrossProblemScaler : public core::IScaler<core::IGrayPenaltyCrossProblem>
	{
		using PtrIGrayPenaltyCrossProblem = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

	public:
		GrayPenaltyCrossProblemScaler(std::shared_ptr<core::IProblemFactory> problem_factory, 
			std::shared_ptr<core::IArrayFactory<float,2>> grayscale_factory,
			std::shared_ptr<core::IArrayFactory<float,3>> color_factory)
			: _problem_factory(problem_factory), _grayscale_factory(grayscale_factory),_color_factory(color_factory)
		{}

		virtual PtrIGrayPenaltyCrossProblem Scale(const PtrIGrayPenaltyCrossProblem input, const size_t& dst_width,
			const size_t& dst_height) override
		{
			auto in_width = input->FirstFrame->Shape[1];
			auto in_height = input->FirstFrame->Shape[0];
			auto color_count = input->CrossFilterImage->Shape[0];

			auto output = _problem_factory->CreateGrayPenaltyCrossProblem();

			std::array<const size_t, 2> gray_shape = { dst_height,dst_width };
			std::array<const size_t, 3> color_shape = { color_count,dst_height,dst_width };

			output->FirstFrame = _grayscale_factory->Zeros(gray_shape);
			output->SecondFrame= _grayscale_factory->Zeros(gray_shape);
			output->CrossFilterImage = _color_factory->Zeros(color_shape);

			if (dst_width < in_width && dst_height < in_height)
			{
				auto in_img = std::dynamic_pointer_cast<Array<float, 2>>(input->FirstFrame);
				auto out_img = std::dynamic_pointer_cast<Array<float, 2>>(output->FirstFrame);
				cpu_backend::_inner::DownScaleGaussianGrayScale<float>(in_img->Data(), in_width, in_height, dst_width, dst_height, out_img->Data());

				in_img = std::dynamic_pointer_cast<Array<float, 2>>(input->SecondFrame);
				out_img = std::dynamic_pointer_cast<Array<float, 2>>(output->SecondFrame);
				cpu_backend::_inner::DownScaleGaussianGrayScale<float>(in_img->Data(), in_width, in_height, dst_width, dst_height, out_img->Data());

				auto in_flow = std::dynamic_pointer_cast<Array<float, 3>>(input->CrossFilterImage);
				auto out_flow = std::dynamic_pointer_cast<Array<float, 3>>(output->CrossFilterImage);
				cpu_backend::_inner::DownScaleGaussianColorScale<float>(in_flow->Data(), in_width, in_height, dst_width, dst_height, color_count, out_flow->Data());

				output->PenaltyFunc = input->PenaltyFunc;
			}
			else
			{
				auto in_img = std::dynamic_pointer_cast<Array<float, 2>>(input->FirstFrame);
				auto out_img = std::dynamic_pointer_cast<Array<float, 2>>(output->FirstFrame);
				cpu_backend::_inner::BicubicGrayScale<float>(in_img->Data(), out_img->Data(), in_width, in_height, dst_width, dst_height);

				in_img = std::dynamic_pointer_cast<Array<float, 2>>(input->SecondFrame);
				out_img = std::dynamic_pointer_cast<Array<float, 2>>(output->SecondFrame);
				cpu_backend::_inner::BicubicGrayScale<float>(in_img->Data(), out_img->Data(), in_width, in_height, dst_width, dst_height);

				auto in_flow = std::dynamic_pointer_cast<Array<float, 3>>(input->CrossFilterImage);
				auto out_flow = std::dynamic_pointer_cast<Array<float, 3>>(output->CrossFilterImage);
				cpu_backend::_inner::BicubicColorScale<float>(in_flow->Data(), out_flow->Data(), in_width, in_height, dst_width, dst_height);

				output->PenaltyFunc = input->PenaltyFunc;
			}
			
			OF_LOG_IMAGE2DARRAY("First Image Scaled", output->FirstFrame);
			OF_LOG_IMAGE2DARRAY("Second Image Scaled", output->SecondFrame);

			return output;
		}


	private:
		std::shared_ptr<core::IProblemFactory> _problem_factory;

		std::shared_ptr<core::IArrayFactory<float, 2>> _grayscale_factory;
		std::shared_ptr<core::IArrayFactory<float, 3>> _color_factory;
	};
}