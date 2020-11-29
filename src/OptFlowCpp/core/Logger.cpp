#pragma once

#include"Logger.h"


namespace core
{
    std::shared_ptr<spdlog::logger> Logger::_Logger;
    bool Logger::_should_log = false;
    void Logger::Init()
    {
        spdlog::set_pattern("%^[%T] %n: %v%$");
        _Logger = spdlog::stdout_color_mt("Optical Flow");
        _Logger->set_level(spdlog::level::trace);
        _should_log = true;

        
    }

    
}