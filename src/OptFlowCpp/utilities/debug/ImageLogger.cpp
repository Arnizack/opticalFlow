#pragma once
#include"ImageLogger.h"
#include"utilities/image_helper/ImageHelper.h"
#include"utilities/flow_helper/FlowHelper.h"

namespace debug
{
    std::string ImageLogger::_image_directory;
    std::string ImageLogger::_flow_directory;

    std::atomic<int> ImageLogger::_image_counter;
    std::atomic<int> ImageLogger::_flow_counter;
    bool ImageLogger::_should_log;

    std::string ImageLogger::GetFlowFilepath(std::string name)
    {
        std::stringstream filepath;
        filepath << _flow_directory << "\\" << name;
        filepath << _flow_counter << ".flo";
        _flow_counter++;
        return filepath.str();
    }
    void ImageLogger::Init(std::string image_directory, std::string flow_directory)
    {
        _image_directory = image_directory;
        _flow_directory = flow_directory;
        _image_counter = 0;
        _flow_counter = 0;
        _should_log = false;
    }
    void ImageLogger::BeginLogging() { _should_log = true; }
    void ImageLogger::EndLogging() { _should_log = false; }
    void ImageLogger::Log2DImage(std::string name, float* data, size_t width, size_t height)
    {
        if (!_should_log) return;
        std::string filepath = GetImageFilepath(name);

        imagehelper::SaveImage(filepath, data, width, height, 1);

    }
    void ImageLogger::Log2DImage(std::string name, std::shared_ptr<core::IArray<float, 2>> image)
    {
        if (!_should_log) return;
        std::string filepath = GetImageFilepath(name);
        imagehelper::SaveImage(filepath, image);
    }
    void ImageLogger::Log3DImage(std::string name, float* data, size_t width, size_t height, size_t color_band)
    {
        if (!_should_log) return;
        std::string filepath = GetImageFilepath(name);
        imagehelper::SaveImage(filepath, data, width, height, color_band);
    }
    void ImageLogger::Log3DImage(std::string name, std::shared_ptr<core::IArray<float, 3>> image)
    {
        if (!_should_log) return;
        std::string filepath = GetImageFilepath(name);
        imagehelper::SaveImage(filepath, image);
    }
    void ImageLogger::LogFlowField(std::string name, double* data, size_t width, size_t height)
    {
        if (!_should_log) return;
        std::string filepath = GetFlowFilepath(name);
        flowhelper::FlowField flow;
        flow.data = std::make_shared<std::vector<double>>(width * height * 2);
        std::copy_n(data, width * height * 2, flow.data->data());
        flow.height = height;
        flow.width = width;
        flowhelper::SaveFlow(filepath, flow);
    }
    void ImageLogger::LogFlowField(std::string name, std::shared_ptr<core::IArray<double, 3>> data)
    {
        if (!_should_log) return;
        std::string filepath = GetFlowFilepath(name);
        flowhelper::SaveFlow(filepath, data);
    }
}