#pragma once


namespace cpu_backend
{
    void BilateralMedianEdgeSpeedup(
		double* flow,
		double* auxiliary_flow,
		double* log_occlusion,
		float* image,
		double filter_influence,
		double auxiliary_influence,
		double sigma_distance,
		double sigma_color,
		bool* is_edge_map,
		int bilateral_filter_length,
        int median_filter_length,
		size_t width,
		size_t height,
		size_t color_channel_count,

		double* destination
	);

    void MedianFilterFlowAt(int x, int y,
        double* flow, int filter_length,
        double* median_list_x,
        double* median_list_y,
		size_t width,
		size_t height,double* destination);
}