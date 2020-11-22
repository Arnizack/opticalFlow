#pragma once
#include <nlohmann/json.hpp>
#include <string>

#include "../JSONHelper.h"
#include "optflow_composition/ContainerOptions.h"
#include "LevelContainerParser.h"

namespace json_settings
{
	namespace modern_json = nlohmann;

	void ParsePyramidResolution(const modern_json::json& input, optflow_composition::PyramidResolution& pyramid_resolution, const std::string string_end = "");

	void ParsePyramid(const modern_json::json& input, optflow_composition::PyramidSettings& pyramid_settings, const std::string string_end = "");

	void ParsePyramidContainerSettings(const  modern_json::json& input, optflow_composition::PyramidIteratorSettings& pyramid_iterator_settings, const std::string string_end = "");

	void GeneratePyramidResolution(modern_json::json& output, optflow_composition::PyramidResolution& pyramid_resolution, const std::string string_end = "");

	void GeneratePyramid(modern_json::json& output, optflow_composition::PyramidSettings& pyramid_settings, const std::string string_end = "");

	void GeneratePyramidContainerSettings(modern_json::json& output, optflow_composition::PyramidIteratorSettings& pyramid_iterator_settings, const std::string string_end = "");
}