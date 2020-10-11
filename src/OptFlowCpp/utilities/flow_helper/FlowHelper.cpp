#pragma once
#include"FlowHelper.h"
#include"libs/imageLib/imageLib/Image.h"
#include"libs/flowcode/flowIO.h"
#include"libs/flowcode/colorcode.h"
#include"utilities/image_helper/ImageHelper.h"

namespace flowhelper
{
    FlowField OpenFlow(std::string filepath)
    {
        CFloatImage flow;
        ReadFlowFile(flow, filepath.data());

        size_t width = flow.Shape().width;
        size_t height = flow.Shape().height;

        FlowField flow_result;
        flow_result.width = width;
        flow_result.height = height;
        flow_result.data =
            std::make_shared<std::vector<double>>(width * height * 2);
        
        double* flow_result_data = flow_result.data->data();
        for (int band = 0; band < 2; band++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    size_t coord = (1-band) * width * height + y * width + x;
                    double val = flow.Pixel(x, y, band);
                    flow_result_data[coord] = val;
                }
            }
            
        }
        return flow_result;
    }

    FlowField IArray2Flow(std::shared_ptr < core::IArray<double, 3>> flow)
    {
        size_t width = flow->Shape[2];
        size_t height = flow->Shape[1];
        size_t size = flow->Size();
        FlowField flow_convert;
        flow_convert.width = width;
        flow_convert.height = height;
        flow_convert.data = std::make_shared<std::vector<double>>(size);

        flow->CopyDataTo(flow_convert.data->data());
        return flow_convert;
        
    }

    void SaveFlow(std::string filepath, FlowField flow)
    {
        size_t width = flow.width;
        size_t height = flow.height;
        CFloatImage flow_convert(width,height,2);

        double* inflow = flow.data->data();

        for (int band = 0; band < 2; band++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    size_t coord = (1-band) * width * height + y * width + x;
                    float val = inflow[coord];
                    flow_convert.Pixel(x, y, band) = val;
                }
            }
        }
        WriteFlowFile(flow_convert, filepath.data());
    }
    void SaveFlow(std::string filepath, 
        std::shared_ptr < core::IArray<double, 3>> flow)
    {
        FlowField flow_convert = IArray2Flow(flow);
        SaveFlow(filepath, flow_convert);
    }
    void SaveFlow2Color(std::string filepath, FlowField flow)
    {
        size_t width = flow.width;
        size_t height = flow.height;
        imagehelper::Image img;
        img.width = width;
        img.height = height;
        img.color_count = 3;
        img.data = std::make_shared<std::vector<float>>(width * height * 3);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < height; x++)
            {
                float flow_x = flow.GetXFlow(x, y);
                float flow_y = flow.GetYFlow(x, y);
                uchar color[3] = {};
                computeColor(flow_x, flow_y, color);
                img.Pixel(x, y, 0) = color[0];
                img.Pixel(x, y, 1) = color[1];
                img.Pixel(x, y, 2) = color[2];
            }
        }
        imagehelper::SaveImage(filepath, img);

    }
    void SaveFlow2Color(std::string filepath, std::shared_ptr<core::IArray<double, 3>> flow)
    {
        FlowField flow_convert =  IArray2Flow(flow);
        SaveFlow2Color(filepath, flow_convert);
    }
    double& FlowField::GetXFlow(size_t x, size_t y)
    {
        return data->operator[](width* height + y * width + x);
    }
    double& FlowField::GetYFlow(size_t x, size_t y)
    {
        return data->operator[]( y * width + x);
    }
}