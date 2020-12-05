#pragma once

namespace cpu_backend
{
	void BilateralMedian(
		double* flow,
		double* auxiliary_flow,
		double* log_occlusion,
		float* image,
		double filter_influence,
		double auxiliary_influence,
		double sigma_distance,
		double sigma_color,
		int filter_length,
		size_t width,
		size_t height,
		size_t color_channel_count,
		double* destination
	);


	//returns the weights count
	size_t BilateralMedianWeight(int x, int y,
		double* log_occlusion,
		float* image,
		size_t width,
		size_t height,
		size_t color_channel_count,
		int filter_length,
		double sigma_distance,
		double sigma_color,
		double* destination);

	void _BilateralMedianList(int x, int y, double* flow, double* auxiliary_flow,
		double filter_influence, double auxiliary_influence, double* weights,
		int filter_length,
		size_t weigths_count, size_t width, size_t height, double* destination);

	void _BilateralMedianListFlowSpecific(int x, int y, double* flow,
		double* auxiliary_flow,
		double filter_influence,
		double auxiliary_influence,
		int filter_length, size_t width, size_t height, double* destination);

	void _BilateralMedianListWeightsSpecific(int x, int y,
		double* auxiliary_flow,
		double filter_influence,
		double auxiliary_influence,
		double* weights,
		int filter_length,
		size_t weigths_count, size_t width, size_t height, double* destination);

	void BilateralMedianAt(int x, int y,
		double* flow,
		double* auxiliary_flow,
		double* log_occlusion,
		float* image,
		double* weights,
		double* median_list_x,
		double* median_list_y,
		double filter_influence,
		double auxiliary_influence,
		double sigma_distance,
		double sigma_color,
		int filter_length,
		size_t width,
		size_t height,
		size_t color_channel_count,
		double* destination);

	namespace _inner
	{
		//length musst be odd
		double _median(double* list, size_t length);
	}
}