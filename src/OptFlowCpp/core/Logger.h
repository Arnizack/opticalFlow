#pragma once
//Inspired by chernos Hazel
#pragma warning(push, 0)
#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"
#pragma warning(pop)
#include<memory>

namespace core
{
	class Logger
	{
	public:
		static void Init();

		static std::shared_ptr<spdlog::logger>& GetLogger() { return _Logger; }
	private:
		static std::shared_ptr<spdlog::logger> _Logger;
	};
}

//log macros
#define OF_LOG_TRACE(...)    ::core::Logger::GetLogger()->trace(__VA_ARGS__)
#define OF_LOG_INFO(...)     ::core::Logger::GetLogger()->info(__VA_ARGS__)
#define OF_LOG_WARN(...)     ::core::Logger::GetLogger()->warn(__VA_ARGS__)
#define OF_LOG_ERROR(...)    ::core::Logger::GetLogger()->error(__VA_ARGS__)
#define OF_LOG_CRITICAL(...) ::core::Logger::GetLogger()->critical(__VA_ARGS__)