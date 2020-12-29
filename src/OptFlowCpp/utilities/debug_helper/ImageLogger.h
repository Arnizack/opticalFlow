#pragma once
#include<memory>
#include<string>
#include <sstream>
#include<chrono>
#include"core/IArray.h"

namespace debug_helper
{
	class ImageLogger
	{
	private:
		static std::string _image_directory;
		static std::string _flow_directory;

		static std::atomic<int> _image_counter;
		static std::atomic<int> _flow_counter;
		static bool _should_log;
		static bool _is_available;

		static std::string GetImageFilepath(std::string name);

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

		static void LogMask(std::string name, bool* data, size_t width, size_t height);
	};
}

//image logger macros
#ifndef OPF_LOG_IMAGE_FLOW_ACTIVATED
	#define OPF_LOG_IMAGE_FLOW_ACTIVATED 0
#endif  //!OPF_LOG_IMAGE_FLOW_ACTIVATED

#if OPF_LOG_IMAGE_FLOW_ACTIVATED
	#define TO_STRING(s) #s
	#define OPF_LOG_IMAGE2D(name,data,width,height) ::debug_helper::ImageLogger::Log2DImage(name ,data,width,height)
	#define OPF_LOG_IMAGE2DARRAY(name,iarray) ::debug_helper::ImageLogger::Log2DImage(name,iarray)
	#define OPF_LOG_IMAGE3D(name,data,width,height,color_band) ::debug_helper::ImageLogger::Log2DImage(name ,data,width,height,color_band)
	#define OPF_LOG_IMAGE3DARRAY(name,iarray) ::debug_helper::ImageLogger::Log2DImage(name,iarray)
	#define OPF_LOG_MASK(name,data,width,height) ::debug_helper::ImageLogger::LogMask(name ,data,width,height)

	#define OPF_LOG_FLOW(name,data,width,height) ::debug_helper::ImageLogger::LogFlowField(name,data,width,height)
	#define OPF_LOG_FLOWARRAY(name,iarray) ::debug_helper::ImageLogger::LogFlowField(name,iarray)

	#define OPF_LOG_IMAGE_FLOW_BEGIN() ::debug_helper::ImageLogger::BeginLogging()
	#define OPF_LOG_IMAGE_FLOW_END() ::debug_helper::ImageLogger::EndLogging()
#else
	#define OPF_LOG_IMAGE2D(name,data,width,height)
	#define OPF_LOG_IMAGE2DARRAY(name,iarray)
	#define OPF_LOG_IMAGE3D(name,data,width,height,color_band)
	#define OPF_LOG_IMAGE3DARRAY(name,iarray)
	#define OPF_LOG_MASK(name,data,width,height)
	#define OPF_LOG_FLOW(name,data,width,height)
	#define OPF_LOG_FLOWARRAY(name,iarray)
	#define OPF_LOG_IMAGE_FLOW_BEGIN()
	#define OPF_LOG_IMAGE_FLOW_END()
#endif