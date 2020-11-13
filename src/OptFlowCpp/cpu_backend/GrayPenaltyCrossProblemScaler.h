#pragma once
#include "core/IScaler.h"
#include "core/solver/problem/IGrayPenaltyCrossProblem.h"
#include "core/solver/problem/IProblemFactory.h"
#include "image/inner/GaussianScale.h"
#include "image/inner/BicubicScale.h"
#include "Array.h"

namespace cpu_backend
{
	class GrayPenaltyCrossProblemScaler : public core::IScaler<core::IGrayPenaltyCrossProblem>
	{
		using PtrIGrayPenaltyCrossProblem = std::shared_ptr<core::IGrayPenaltyCrossProblem>;

	public:
		GrayPenaltyCrossProblemScaler(std::shared_ptr<core::IProblemFactory> problem_factory)
			: _problem_factory(problem_factory)
		{}

		virtual PtrIGrayPenaltyCrossProblem Scale(const PtrIGrayPenaltyCrossProblem input, const size_t& dst_width,
			const size_t& dst_height) override
		{
			auto in_width = input->FirstFrame->Shape[0];
			auto in_height = input->FirstFrame->Shape[1];

			auto output = _problem_factory->CreateGrayPenaltyCrossProblem();

			if (dst_width < in_width)
			{
				auto in_img = std::dynamic_pointer_cast<Array<float, 2>>(input->FirstFrame);
				auto out_img = std::dynamic_pointer_cast<Array<float, 2>>(output->FirstFrame);
				cpu_backend::_inner::Gaussian2DScale(in_img->Data(), out_img->Data(), in_width, in_height, dst_width, dst_height);

				in_img = std::dynamic_pointer_cast<Array<float, 2>>(input->SecondFrame);
				out_img = std::dynamic_pointer_cast<Array<float, 2>>(output->SecondFrame);
				cpu_backend::_inner::Gaussian2DScale(in_img->Data(), out_img->Data(), in_width, in_height, dst_width, dst_height);

				auto in_flow = std::dynamic_pointer_cast<Array<float, 3>>(input->CrossFilterImage);
				auto out_flow = std::dynamic_pointer_cast<Array<float, 3>>(output->CrossFilterImage);
				cpu_backend::_inner::GaussianFlowScale(in_flow->Data(), out_flow->Data(), in_width, in_height, dst_width, dst_height);

				output->PenaltyFunc = input->PenaltyFunc;
			}
			else
			{
				auto in_img = std::dynamic_pointer_cast<Array<float, 2>>(input->FirstFrame);
				auto out_img = std::dynamic_pointer_cast<Array<float, 2>>(output->FirstFrame);
				cpu_backend::_inner::Bicubic2DScale(in_img->Data(), out_img->Data(), in_width, in_height, dst_width, dst_height);

				in_img = std::dynamic_pointer_cast<Array<float, 2>>(input->SecondFrame);
				out_img = std::dynamic_pointer_cast<Array<float, 2>>(output->SecondFrame);
				cpu_backend::_inner::Bicubic2DScale(in_img->Data(), out_img->Data(), in_width, in_height, dst_width, dst_height);

				auto in_flow = std::dynamic_pointer_cast<Array<float, 3>>(input->CrossFilterImage);
				auto out_flow = std::dynamic_pointer_cast<Array<float, 3>>(output->CrossFilterImage);
				cpu_backend::_inner::BicubicFlowScale(in_flow->Data(), out_flow->Data(), in_width, in_height, dst_width, dst_height);

				output->PenaltyFunc = input->PenaltyFunc;
			}

			return output;
		}


	private:
		std::shared_ptr<core::IProblemFactory> _problem_factory;
	};
}