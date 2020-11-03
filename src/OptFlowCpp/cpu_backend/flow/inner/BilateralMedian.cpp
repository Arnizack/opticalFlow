#pragma once
#include"pch.h"
#include"BilateralMedian.h"
#include<algorithm>
#include"../../image/inner/IteratorHelper.h"
#include"../../image/inner/ArrayHelper.h"

namespace cpu_backend
{
    void BilateralMedian(
        double* flow, double* auxiliary_flow, double* log_occlusion, float* image, 
        double filter_influence, double auxiliary_influence, double sigma_distance, 
        double sigma_color, int filter_length, size_t width, size_t height, size_t color_channel_count,
        double* destination)
    {
        size_t weights_max_length = filter_length * filter_length ;
        size_t median_list_max_length = weights_max_length * 2 + 1;
        for (int y = 0; y < height; y++)
        {
            double* weights = (double*) malloc(weights_max_length * sizeof(double));
            double* median_list = (double*)malloc(median_list_max_length * sizeof(double));
            for (int x = 0; x < width; x++)
            {
                int coord = y * width + x;
                BilateralMedianAt(x, y, flow, auxiliary_flow, log_occlusion, image, weights,
                    median_list, filter_influence, auxiliary_influence, sigma_distance, sigma_color, filter_length,
                    width, height, color_channel_count,destination);

            }
            free(weights);
            free(median_list);
        }
    }

    size_t BilateralMedianWeight(int x, int y,
        double* log_occlusion, 
        float* image, 
        size_t width, size_t height, size_t color_channel_count, 
        int filter_length, 
        double sigma_distance, double sigma_color, double* destination)
    {
        //See A Quantitative Analysis of Current Practices in Optical Flow Estimation
        //and the Principles behind Them
        //EQ. 10
        int filter_length_half = filter_length / 2;
        size_t x_start = std::max(x - filter_length_half, 0);
        size_t x_end = std::min<size_t>(x + filter_length_half+1, width );
        size_t y_start = std::max(y - filter_length_half, 0);
        size_t y_end = std::min<size_t>(y + filter_length_half+1, height );
        size_t first_pixel_idx = y * width + x;

        double log_occlusion_current = log_occlusion[first_pixel_idx];

        size_t counter = 0;

        _inner::Iterate2D(x_start, x_end, y_start, y_end, [&](size_t x_compared, size_t y_compared)
        {
            int x_diff = (int)x - (int)x_compared;
            int y_diff = (int)y - (int)y_compared;
            double distance_norm_squared = x_diff * x_diff + y_diff * y_diff;
            size_t second_pixel_idx = y_compared * width + x_compared;

            float color_norm_squared = _inner::ComputeColorDifferenceSquaredNorm(
                image, image, first_pixel_idx, second_pixel_idx, 
                width, height, color_channel_count);


            double log_occlusion_compared = log_occlusion[second_pixel_idx];
            
            double exponent = distance_norm_squared / (2 * sigma_distance * sigma_distance);
            exponent += color_norm_squared / (2 * sigma_color * sigma_color * color_channel_count);
            exponent -= log_occlusion_compared;
            exponent += log_occlusion_current;


            destination[counter] = exp(-exponent);

            counter++;
        });

        return counter;
    }
    
    void _BilateralMedianList(int x, int y, double* flow, 
        double* auxiliary_flow, 
        double filter_influence, 
        double auxiliary_influence, 
        double* weights, 
        int filter_length,
        size_t weigths_count, size_t width, size_t height, double* destination)
    {
        //See A NEW MEDIAN FORMULA WITH APPLICATIONS TO PDE BASED DENOISING
        //EQ. 3.13
        int filter_length_half = filter_length / 2;
        size_t x_start = std::max(x - filter_length_half, 0);
        size_t x_end = std::min<size_t>(x + filter_length_half+1, width );
        size_t y_start = std::max(y - filter_length_half, 0);
        size_t y_end = std::min<size_t>(y + filter_length_half+1, height );
        
        size_t counter = 0;
        _inner::Iterate2D(x_start, x_end, y_start, y_end, [&](size_t x_compared, size_t y_compared)
        {
            destination[counter] = _inner::GetValueAt<double, Padding::ZEROS>(x_compared, y_compared, width, height, flow);
            counter++;
        });
        
        double lambda = auxiliary_influence / filter_influence;
        double multiplikator = 1 / ( 2 * lambda );

        for (int negative_weight_count = 0; negative_weight_count <= weigths_count; negative_weight_count++)
        {
            double sum = 0;
            for (int positiv_weight_idx = weigths_count - 1; positiv_weight_idx >= negative_weight_count; positiv_weight_idx--)
            {
                sum += weights[positiv_weight_idx];
            }
            for (int negativ_weight_idx = negative_weight_count - 1; negativ_weight_idx >= 0; negativ_weight_idx--)
            {
                sum -= weights[negativ_weight_idx];
            }
            destination[counter + negative_weight_count] = multiplikator * sum;
        }

    }
    
    void BilateralMedianAt(
        int x, int y,
        double* flow,
        double* auxiliary_flow,
        double* log_occlusion,
        float* image,
        double* weights,
        double* median_list,
        double filter_influence,
        double auxiliary_influence,
        double sigma_distance,
        double sigma_color,
        int filter_length,
        size_t width,
        size_t height,
        size_t color_channel_count,
        double* destination
    )
    {
        size_t weights_count = BilateralMedianWeight(x, y, log_occlusion, image, width, height, color_channel_count,
            filter_length, sigma_distance, sigma_color, weights);
        //Y Flow
        double* flow_y = flow;
        _BilateralMedianList(x, y, flow, auxiliary_flow, filter_influence, auxiliary_influence, weights, filter_length,
            weights_count, width, height, median_list);
        destination[y*width+x] = _inner::_median(median_list, weights_count * 2 + 1);

        //X Flow
        double* flow_x = flow + width * height;
        _BilateralMedianList(x, y, flow_x, auxiliary_flow, filter_influence, auxiliary_influence, weights, filter_length,
            weights_count, width, height, median_list);
        destination[width*height+y * width + x] = _inner::_median(median_list, weights_count * 2 + 1);

    }

    double _inner::_median(double* list, size_t length)
    {
        double* ptr_middel = list + (length / 2);
        std::nth_element(list, ptr_middel, list + length);
        return *(ptr_middel);
    }
}


