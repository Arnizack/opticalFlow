#pragma once
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

#include "optflow_composition/ContainerOptions.h"

#include "../../JSONHelper.h"

namespace json_settings
{
	namespace modern_json = nlohmann;

	void ParseCrossMedianFilter(const modern_json::json& input, std::shared_ptr<cpu_backend::CrossMedianFilterSettings> cross_median_filter, const std::string string_end = "");

	void GenerateCrossMedianFilter(modern_json::json& output, std::shared_ptr<cpu_backend::CrossMedianFilterSettings> cross_median_filter, const std::string string_end = "");
}