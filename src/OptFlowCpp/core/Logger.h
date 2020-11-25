#pragma once
//Inspired by chernos Hazel
#pragma warning(push, 0)
#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"
#pragma warning(pop)


namespace core
{
	class Logger
	{
	public:
		static void Init();

		static std::shared_ptr<spdlog::logger>& GetLogger() { return _Logger; }
		static bool _should_log;
	private:
		
		static std::shared_ptr<spdlog::logger> _Logger;
	};


	
}

//log macros
#define OF_LOG_TRACE(...)    if(::core::Logger::_should_log) ::core::Logger::GetLogger()->trace(__VA_ARGS__)
#define OF_LOG_INFO(...)     if(::core::Logger::_should_log) ::core::Logger::GetLogger()->info(__VA_ARGS__)
#define OF_LOG_WARN(...)     if(::core::Logger::_should_log) ::core::Logger::GetLogger()->warn(__VA_ARGS__)
#define OF_LOG_ERROR(...)    if(::core::Logger::_should_log) ::core::Logger::GetLogger()->error(__VA_ARGS__)
#define OF_LOG_CRITICAL(...) if(::core::Logger::_should_log) ::core::Logger::GetLogger()->critical(__VA_ARGS__)



