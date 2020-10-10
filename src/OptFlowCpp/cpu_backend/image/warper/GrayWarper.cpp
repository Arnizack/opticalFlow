#pragma once
#include"pch.h"
#include"GrayWarper.h"
#include"InterpolateBicubic.h"

namespace cpu_backend
{
    using PtrGrayImg = std::shared_ptr<core::IArray<float, 2>>;
    using PtrFlowField = std::shared_ptr<core::IArray<double, 3>>;

    void GrayWarper::SetImage(PtrGrayImg image)
    {
        
        auto cast_image = std::static_pointer_cast<Array<float, 2>, core::IArray<float, 2>>(image);
        
        _lookup = CreateBicubicLookup(cast_image);
    }
    PtrGrayImg GrayWarper::Warp(PtrFlowField flow)
    {
        auto ptr_flow = std::static_pointer_cast<Array<double, 3>, core::IArray<double, 3>>(flow);
        size_t width = flow->Shape[1];
        size_t height = flow->Shape[0];

        auto ptr_image = std::make_shared<Array<float, 2>>(std::array<const size_t,2>({height,width}));
        
        float* img_data = ptr_image->Data();

        double* y_flow_data = ptr_flow->Data();

        size_t x_offset = width * height;

        double* x_flow_data = y_flow_data + x_offset;


        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double coord_x = (double)x + x_flow_data[width*y+x];
                double coord_y = (double)y + y_flow_data[width * y + x];

                coord_x = fmin(coord_x, width);
                coord_x = fmax(coord_x, 0);
                coord_y = fmin(coord_y, height);
                coord_y = fmax(coord_y, 0);

                size_t corner_x = fmin(coord_x, width - 1);
                size_t corner_y = fmin(coord_y, height - 1);

                float* scalars = _lookup->operator[](corner_x* width + corner_y).data();

                double remap_coord_x = coord_x - corner_x;
                double remap_coord_y = coord_y - corner_y;

                float interpol_val = BicubicApplyScalars(scalars, remap_coord_x, remap_coord_y);

                img_data[y * width + x] = interpol_val;

            }
        }

        return ptr_image;
    }

    std::unique_ptr<std::vector<std::array<float, 16>>> GrayWarper::CreateBicubicLookup(std::shared_ptr<Array<float, 2>> image)
    {
        int width = image->Shape[1];
        int height = image->Shape[0];
        auto ptr_img_deriv_x = std::make_unique<std::vector<float>>(width * height);
        auto ptr_img_deriv_y = std::make_unique<std::vector<float>>(width * height);
        auto ptr_img_deriv_xy = std::make_unique<std::vector<float>>(width * height);
        
        _derivative_calculator->ComputeDerivativeX(image->Data(), width, height, ptr_img_deriv_x->data());
        _derivative_calculator->ComputeDerivativeY(image->Data(), width, height, ptr_img_deriv_y->data());
        _derivative_calculator->ComputeDerivativeX(ptr_img_deriv_y->data(), width, height, 
            ptr_img_deriv_xy->data());

        auto lookup_table = std::make_unique<std::vector<std::array<float, 16>>>((width-1)*(height-1));

        float* img_data = image->Data();
        float* deriv_x = ptr_img_deriv_x->data();
        float* deriv_y = ptr_img_deriv_y->data();
        float* deriv_xy = ptr_img_deriv_xy->data();

        for (int y = 0; y < height-1; y++)
        {
            for (int x = 0; x < width - 1; x++)
            {
                int lower_left_coord = width * y + x;
                int lower_right_coord = width * y + x + 1;
                int upper_left_coord = width * (y + 1) + x;
                int upper_right_coord = width * (y + 1) + x + 1;

                float lower_left_corner  = img_data[lower_left_coord];
                float lower_right_corner = img_data[lower_right_coord];
                float upper_left_corner  = img_data[upper_left_coord];
                float upper_right_corner = img_data[upper_right_coord];

                float lower_left_corner_deriv_x  = deriv_x[lower_left_coord];
                float lower_right_corner_deriv_x = deriv_x[lower_right_coord];
                float upper_left_corner_deriv_x =  deriv_x[upper_left_coord];
                float upper_right_corner_deriv_x = deriv_x[upper_right_coord];

                float lower_left_corner_deriv_y  = deriv_y[lower_left_coord];
                float lower_right_corner_deriv_y = deriv_y[lower_right_coord];
                float upper_left_corner_deriv_y =  deriv_y[upper_left_coord];
                float upper_right_corner_deriv_y = deriv_y[upper_right_coord];

                float lower_left_corner_deriv_xy  = deriv_xy[lower_left_coord];
                float lower_right_corner_deriv_xy = deriv_xy[lower_right_coord];
                float upper_left_corner_deriv_xy =  deriv_xy[upper_left_coord];
                float upper_right_corner_deriv_xy = deriv_xy[upper_right_coord];

                float* dst = lookup_table->operator[](lower_left_coord).data();

                InterpolateBicubicScalars(lower_left_corner, lower_right_corner,
                    upper_left_corner, upper_right_corner,
                    lower_left_corner_deriv_x, lower_right_corner_deriv_x,
                    upper_left_corner_deriv_x, upper_right_corner_deriv_x,
                    lower_left_corner_deriv_y, lower_right_corner_deriv_y,
                    upper_left_corner_deriv_y, upper_right_corner_deriv_y,
                    lower_left_corner_deriv_xy, lower_right_corner_deriv_xy,
                    upper_left_corner_deriv_xy, upper_right_corner_deriv_xy, dst);

            }
        }
        return lookup_table;

        
    }
}