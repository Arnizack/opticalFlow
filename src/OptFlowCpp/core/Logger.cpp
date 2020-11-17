#pragma once
#include"pch.h"
#include"Logger.h"

namespace core
{
    std::shared_ptr<spdlog::logger> Logger::_Logger;
    void Logger::Init()
    {
        spdlog::set_pattern("%^[%T] %n: %v%$");
        _Logger = spdlog::stdout_color_mt("Optical Flow");
        _Logger->set_level(spdlog::level::trace);

        
    }
}