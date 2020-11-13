#pragma once
#include"pch.h"
#include"ConvertToGrayscale.h"
#include"ArrayHelper.h"

namespace cpu_backend
{
    void ConvertColorToGrayscale(float* image, size_t width, size_t height, float* destination)
    {
        float* image_red = image;
        float* image_green = image + width * height;
        float* image_blue = image + width * height * 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float red = _inner::GetValueAt<float,Padding::ZEROS>(x, y, (int) width, (int) height, image_red);
                float green = _inner::GetValueAt<float,Padding::ZEROS>(x, y, (int) width, (int) height, image_green);
                float blue = _inner::GetValueAt<float, Padding::ZEROS>(x, y, (int)width, (int)height, image_blue);
                destination[y*width+x] = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
            }
        }
    }
}