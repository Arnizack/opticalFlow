#pragma once

#include"LogOcclusion.h"
#include<algorithm>
#include"../../image/inner/ArrayHelper.h"

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
		double sigma_div, double sigma_error 
	)
	{
		//See EQ. 11. A Quantitative Analysis of Current Practices in Optical Flow Estimation
		//and the Principles behind Them
 
		int pixel_count = width * height;
		#pragma omp parallel for
		for (int i = 0; i < pixel_count; i++)
		{
			double d = std::min(flow_div[i], 0.0);
			double exponent = d * d / (2.0 * sigma_div * sigma_div);
			double color_difference_norm = _inner::ComputeColorDifferenceSquaredNorm(image, image_warped,i,i, width, height, color_channels_count);
			
			exponent += color_difference_norm * color_difference_norm / (2 * sigma_error * sigma_error);
			destination[i] = -exponent;
		}
	}
}