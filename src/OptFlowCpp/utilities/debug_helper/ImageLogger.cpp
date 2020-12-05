#pragma once
#include"ImageLogger.h"
#include"image_helper/ImageHelper.h"
#include"flow_helper/FlowHelper.h"

#if _WIN32
    #include <windows.h>
    #define MAKE_DIR(path) CreateDirectory(path, NULL)
#elif __linux__
    #include <sys/stat.h>
    #define MAKE_DIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#else
#endif



namespace debug_helper
{
    std::string ImageLogger::_image_directory;
    std::string ImageLogger::_flow_directory;

    std::atomic<int> ImageLogger::_image_counter;
    std::atomic<int> ImageLogger::_flow_counter;
    bool ImageLogger::_should_log = false;
    bool ImageLogger::_is_available = false;

    inline std::string ImageLogger::GetImageFilepath(std::string name)
    {
        std::stringstream filepath;
        filepath << _image_directory << "\\" ;
        filepath << _image_counter << " " << name << ".png";
        _image_counter++;
        return filepath.str();
    }

    std::string ImageLogger::GetFlowFilepath(std::string name)
    {
        std::stringstream filepath;
        filepath << _flow_directory << "\\";
        filepath << _flow_counter << " " << name << ".png";
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
        _is_available = true;
        MAKE_DIR(image_directory.c_str());
        MAKE_DIR(flow_directory.c_str());
    }
    void ImageLogger::BeginLogging() { _should_log = _is_available; }
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
        flowhelper::SaveFlow2Color(filepath, flow);
    }
    void ImageLogger::LogFlowField(std::string name, std::shared_ptr<core::IArray<double, 3>> data)
    {
        if (!_should_log) return;
        std::string filepath = GetFlowFilepath(name);
        flowhelper::SaveFlow2Color(filepath, data);
    }
    void ImageLogger::LogMask(std::string name, bool* data, size_t width, size_t height)
    {
        if (!_should_log) return;
        std::string filepath = GetImageFilepath(name);
        auto img = std::make_unique<std::vector<float>>(width * height);
        float* img_data = img->data();
        for(int i = 0; i< width*height;i++ )
        {
            if (data[i])
                img_data[i] = 1;
            else
                img_data[i] = 0;
        }
        
        imagehelper::SaveImage(filepath, img_data, width, height, 1);
    }
}