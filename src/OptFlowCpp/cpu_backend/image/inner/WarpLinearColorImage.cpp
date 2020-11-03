#pragma once
#include"pch.h"
#include"WarpLinearColorImage.h"
#include"IteratorHelper.h"
#include"BilinearInterpolate.h"

namespace cpu_backend
{
    void WarpLinearColorImage(float* destination, float* image, double* flow, size_t width, size_t height, size_t color_channel_count)
    {
        double* flow_y = flow;
        double* flow_x = flow + width * height;
        _inner::Iterate2D<size_t>(0, width, 0, height,
            [&](size_t x, size_t y)
        {
            size_t coord = y * width + x;
            double shift_x = flow_x[coord];
            double shift_y = flow_y[coord];

            double shifted_x_coord = (double)x + shift_x;
            double shifted_y_coord = (double)y + shift_y;
            for(size_t color_channel_idx = 0; color_channel_idx < color_channel_count; color_channel_idx++)
            {
                float* channel_image = image + color_channel_idx * (width * height);
                float result = BilinearInterpolateAt(shifted_x_coord, shifted_y_coord, channel_image, width, height);
                destination[coord + color_channel_idx * (width * height) ] = result;
            }
        });
        
        
    }
}