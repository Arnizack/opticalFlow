#pragma once
#include <nlohmann/json.hpp>
#include <memory>

#include "optflow_composition/CPUBackendOptions.h"

#include "CrossMedianFilterParser.h"
#include "CharbonnierPenaltyParser.h"
#include "LinearSystemParser.h"

namespace json_settings
{
	namespace modern_json = nlohmann;

	void ParseCpu(const modern_json::json& input, std::shared_ptr<optflow_composition::CPUBackendOptions> cpu_backend, std::string string_end = "");

	void GenerateCpu(modern_json::json& output, std::shared_ptr<optflow_composition::CPUBackendOptions> cpu_backend, std::string string_end = "");
}