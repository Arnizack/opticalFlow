#pragma once
#include"CharbonnierPenaltyParser.h"

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName(var) (#var)

namespace json_settings
{
    void ParseCharbonnierPenalty(const modern_json::json& input, std::shared_ptr<cpu_backend::CharbonnierPenaltySettings> charbonnier_penalty, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(charbonnier_penalty->DefaultBlendFactor), string_end);
        if (input.count(option_name))
            charbonnier_penalty->DefaultBlendFactor = input[option_name];

        option_name = GetOptionName(VarName(charbonnier_penalty->Epsilon), string_end);
        if (input.count(option_name))
            charbonnier_penalty->Epsilon = input[option_name];

        option_name = GetOptionName(VarName(charbonnier_penalty->Exponent), string_end);
        if (input.count(option_name))
            charbonnier_penalty->Exponent = input[option_name];
    }

    void GenerateCharbonnierPenalty(modern_json::json& output, std::shared_ptr<cpu_backend::CharbonnierPenaltySettings> charbonnier_penalty, const std::string string_end)
    {
        std::string option_name = GetOptionName(VarName(charbonnier_penalty->DefaultBlendFactor), string_end);
        output[option_name] = charbonnier_penalty->DefaultBlendFactor;

        option_name = GetOptionName(VarName(charbonnier_penalty->Epsilon), string_end);
        output[option_name] = charbonnier_penalty->Epsilon;

        option_name = GetOptionName(VarName(charbonnier_penalty->Exponent), string_end);
        output[option_name] = charbonnier_penalty->Exponent;
    }
}