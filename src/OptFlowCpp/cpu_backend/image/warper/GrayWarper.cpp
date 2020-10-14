#pragma once
#include"pch.h"
#include<omp.h>
#include"GrayWarper.h"
#include"../inner/BicubicInterpolate.h"

namespace cpu_backend
{
    using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
    using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;

    

    
    void GrayWarper::SetImage(PtrGrayImg image)
    {
        _image = std::static_pointer_cast<Array<float, 2>, core::IArray<float, 2>>(image);
    }

    PtrGrayImg GrayWarper::Warp(PtrFlowField flow)
    {
        size_t width = flow->Shape[2];
        size_t height = flow->Shape[1];
        auto flow_real = std::static_pointer_cast<Array<double, 3>, core::IArray<double, 3>>(flow);

        float* img_data = _image->Data();

        std::array<const size_t, 2>&& shape = { height,width };

        auto result_img = std::make_shared<Array<float, 2>>(shape);

        #pragma omp parallel for
        for (int y = 0; y < height; y++)
        {
            for (size_t x = 0; x < width; x++)
            {
                float flow_x = (*flow_real)[{1, (size_t)y, x}];
                float flow_y = (*flow_real)[{0, (size_t)y, x}];
                float x_inter = (float)x + flow_x;
                float y_inter = (float)y + flow_y;

                float interpolated = _inner::BicubicInerpolateAt<float>(x_inter, y_inter, img_data, width, height);
                (*result_img)[{(size_t)y, x}] = interpolated;
            
            }
        }
        return std::static_pointer_cast<core::IArray<float, 2>, Array<float, 2>>(result_img);
    }

}