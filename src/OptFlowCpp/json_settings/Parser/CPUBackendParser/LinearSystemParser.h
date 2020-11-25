#pragma once
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

#include "optflow_composition/ContainerOptions.h"

#include "../../JSONHelper.h"

namespace json_settings
{
	namespace modern_json = nlohmann;

	void ParseLinearSystem(const modern_json::json& input, std::shared_ptr<cpu_backend::LinearSystemSettings> linear_system, const std::string string_end = "");

	void GenerateLinearSystem(modern_json::json& output, std::shared_ptr<cpu_backend::LinearSystemSettings> linear_system, const std::string string_end = "");
}