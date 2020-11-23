#pragma once
#include <nlohmann/json.hpp>
#include <memory>

#include "optflow_composition/CPUBackendOptions.h"

namespace console_ui
{
	namespace modern_json = nlohmann;

	void ParseCpu(const modern_json::json& input, std::shared_ptr<optflow_composition::CPUBackendOptions> cpu_backend, std::string string_end = "");

	void GenerateCpu(const modern_json::json& output, std::shared_ptr<optflow_composition::CPUBackendOptions> cpu_backend, std::string string_end = "");
}