#pragma once
#include"CrossMedianFilterParser.h"

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName(var) (#var)

namespace json_settings
{
    void ParseCrossMedianFilter(const modern_json::json& input, std::shared_ptr<cpu_backend::CrossMedianFilterSettings> cross_median_filter, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(cross_median_filter->FilterInfluence), string_end);
        if (input.count(option_name))
            cross_median_filter->FilterInfluence = input[option_name];

        option_name = GetOptionName(VarName(cross_median_filter->FilterLength), string_end);
        if (input.count(option_name))
            cross_median_filter->FilterLength = input[option_name];

        option_name = GetOptionName(VarName(cross_median_filter->SigmaColor), string_end);
        if (input.count(option_name))
            cross_median_filter->SigmaColor = input[option_name];

        option_name = GetOptionName(VarName(cross_median_filter->SigmaDistance), string_end);
        if (input.count(option_name))
            cross_median_filter->SigmaDistance = input[option_name];

        option_name = GetOptionName(VarName(cross_median_filter->SigmaDiv), string_end);
        if (input.count(option_name))
            cross_median_filter->SigmaDiv = input[option_name];

        option_name = GetOptionName(VarName(cross_median_filter->SigmaError), string_end);
        if (input.count(option_name))
            cross_median_filter->SigmaError = input[option_name];
    }

    void GenerateCrossMedianFilter(modern_json::json& output, std::shared_ptr<cpu_backend::CrossMedianFilterSettings> cross_median_filter, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(cross_median_filter->FilterInfluence), string_end);
        output[option_name] = cross_median_filter->FilterInfluence;

        option_name = GetOptionName(VarName(cross_median_filter->FilterLength), string_end);
        output[option_name] = cross_median_filter->FilterLength;

        option_name = GetOptionName(VarName(cross_median_filter->SigmaColor), string_end);
        output[option_name] = cross_median_filter->SigmaColor;

        option_name = GetOptionName(VarName(cross_median_filter->SigmaDistance), string_end);
        output[option_name] = cross_median_filter->SigmaDistance;

        option_name = GetOptionName(VarName(cross_median_filter->SigmaDiv), string_end);
        output[option_name] = cross_median_filter->SigmaDiv;

        option_name = GetOptionName(VarName(cross_median_filter->SigmaError), string_end);
        output[option_name] = cross_median_filter->SigmaError;
    }
}