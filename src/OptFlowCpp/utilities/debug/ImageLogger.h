#pragma once
#include<memory>
#include<string>
#include <sstream>
#include<chrono>
#include"core/IArray.h"

namespace debug
{
	class ImageLogger
	{
	private:
		static std::string _image_directory;
		static std::string _flow_directory;

		static std::atomic<int> _image_counter;
		static std::atomic<int> _flow_counter;
		static bool _should_log;

		static std::string GetImageFilepath(std::string name)
		{
			std::stringstream filepath;
			filepath << _image_directory << "\\" << name;
			filepath << _image_counter << ".png";
			_image_counter++;
			return filepath.str();
		}

		static std::string GetFlowFilepath(std::string name);

	public:
		static void Init(std::string image_directory, std::string flow_directory);;

		static void BeginLogging();;
		static void EndLogging();

		static void Log2DImage(std::string name, float* data, size_t width, size_t height);
		static void Log2DImage(std::string name, std::shared_ptr<core::IArray<float, 2>> image);

		static void Log3DImage(std::string name, float* data, size_t width, size_t height, size_t color_band);
		static void Log3DImage(std::string name, std::shared_ptr<core::IArray<float, 3>> image);
		static void LogFlowField(std::string name, double* data, size_t width, size_t height);

		static void LogFlowField(std::string name, std::shared_ptr<core::IArray<double, 3>> data);
	};
}

//image logger macros

#define OF_LOG_IMAGE_AND_FLOW 0
#define TO_STRING(s) #s
#define OF_LOG_IMAGE2D(name,data,width,height) ::debug::ImageLogger::Log2DImage(name ,data,width,height)
#define OF_LOG_IMAGE2DARRAY(name,iarray) ::debug::ImageLogger::Log2DImage(name,iarray)
#define OF_LOG_IMAGE3D(name,data,width,height,color_band) ::debug::ImageLogger::Log2DImage(name ,data,width,height,color_band)
#define OF_LOG_IMAGE3DARRAY(name,iarray) ::debug::ImageLogger::Log2DImage(name,iarray)

#define OF_LOG_FLOW(name,data,width,height) ::debug::ImageLogger::LogFlowField(name,data,width,height)
#define OF_LOG_FLOWARRAY(name,iarray) ::debug::ImageLogger::LogFlowField(name,iarray)

#define OF_LOG_IMAGE_FLOW_BEGIN() ::debug::ImageLogger::BeginLogging()
#define OF_LOG_IMAGE_FLOW_END() ::debug::ImageLogger::EndLogging()