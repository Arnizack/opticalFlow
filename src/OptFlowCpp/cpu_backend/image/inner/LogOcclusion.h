#pragma once
#include"../warper/GrayWarper.h"
namespace cpu_backend
{
	void ComputeLogOcclusion(
		double* destination,
		float* image,
		float* image_warped,
		double* flow_div,
		size_t width,
		size_t height,
		size_t color_channels_count,
		double sigma_div = 0.3,
		double sigma_error = 20
	);
}