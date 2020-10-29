#pragma once

namespace cpu_backend
{
	void WarpLinearColorImage(float* destination, float* image, double* flow,
		size_t width, size_t height, size_t color_channel_count);
}