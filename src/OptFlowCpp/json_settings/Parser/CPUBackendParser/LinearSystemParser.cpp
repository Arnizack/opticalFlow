#pragma once
#include"LinearSystemParser.h"

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName(var) (#var)

namespace json_settings
{
    void ParseLinearSystem(const modern_json::json& input, std::shared_ptr<cpu_backend::LinearSystemSettings> linear_system, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(linear_system->LambdaKernel), string_end);
        if (input.count(option_name))
            linear_system->LambdaKernel = input[option_name];
    }

    void GenerateLinearSystem(modern_json::json& output, std::shared_ptr<cpu_backend::LinearSystemSettings> linear_system, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(linear_system->LambdaKernel), string_end);
        output[option_name] = linear_system->LambdaKernel;
    }
}