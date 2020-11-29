#pragma once
#include"CPUBackendParser.h"

namespace json_settings
{
    void ParseCpu(const modern_json::json& input, std::shared_ptr<optflow_composition::CPUBackendOptions> cpu_backend, std::string string_end)
    {
        ParseCharbonnierPenalty(input, cpu_backend->CharbonnierPenalty, string_end);
        ParseCrossMedianFilter(input, cpu_backend->CrossMedianFilter, string_end);
        ParseLinearSystem(input, cpu_backend->LinearSystem, string_end);
    }

    void GenerateCpu(modern_json::json& output, std::shared_ptr<optflow_composition::CPUBackendOptions> cpu_backend, std::string string_end)
    {
        GenerateCharbonnierPenalty(output, cpu_backend->CharbonnierPenalty, string_end);
        GenerateCrossMedianFilter(output, cpu_backend->CrossMedianFilter, string_end);
        GenerateLinearSystem(output, cpu_backend->LinearSystem, string_end);
    }
}