#pragma once
#include"PyramidContainerParser.h"

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName

namespace console_ui
{
	void ParsePyramidResolution(const modern_json::json& input, optflow_composition::PyramidResolution& pyramid_resolution, const std::string string_end)
	{
		std::string option_name = GetOptionName(VarName(pyramid_resolution.ScaleFactor), string_end);
		if(input.count(option_name))
			pyramid_resolution.ScaleFactor = input[option_name];

		option_name = GetOptionName(VarName(pyramid_resolution.MinResolutionX), string_end);
		if (input.count(option_name))
			pyramid_resolution.MinResolutionX = input[option_name];

		option_name = GetOptionName(VarName(pyramid_resolution.MinResolutionY), string_end);
		if (input.count(option_name))
			pyramid_resolution.MinResolutionY = input[option_name];
	}

	void ParsePyramid(const modern_json::json& input, optflow_composition::PyramidSettings& pyramid_settings, const std::string string_end)
	{
		ParsePyramidResolution(input, pyramid_settings.Resolution, string_end);
		ParseLevelSettings(input, pyramid_settings.SingleLevelSettings, string_end);
	}

	void ParsePyramidContainerSettings(const modern_json::json& input, optflow_composition::PyramidIteratorSettings& pyramid_iterator_settings, const std::string string_end)
	{
		ParsePyramid(input, pyramid_iterator_settings.ConvexSettings, "Convex" + string_end);
		ParsePyramid(input, pyramid_iterator_settings.NonConvexSettings, "Non_Convex" + string_end);
	}

	void GeneratePyramidResolution(modern_json::json& output, optflow_composition::PyramidResolution& pyramid_resolution, const std::string string_end)
	{
		std::string option_name = GetOptionName(VarName(pyramid_resolution.ScaleFactor), string_end);
		output[option_name] = pyramid_resolution.ScaleFactor;

		option_name = GetOptionName(VarName(pyramid_resolution.MinResolutionX), string_end);
		output[option_name] = pyramid_resolution.MinResolutionX;

		option_name = GetOptionName(VarName(pyramid_resolution.MinResolutionY), string_end);
		output[option_name] = pyramid_resolution.MinResolutionY;
	}

	void GeneratePyramid(modern_json::json& output, optflow_composition::PyramidSettings& pyramid_settings, const std::string string_end)
	{
		GeneratePyramidResolution(output, pyramid_settings.Resolution, string_end);
		GenerateLevelSettings(output, pyramid_settings.SingleLevelSettings, string_end);
	}

	void GeneratePyramidContainerSettings(modern_json::json& output, optflow_composition::PyramidIteratorSettings& pyramid_iterator_settings, std::string string_end)
	{
		GeneratePyramid(output, pyramid_iterator_settings.ConvexSettings, "Convex" + string_end);
		GeneratePyramid(output, pyramid_iterator_settings.NonConvexSettings, "Non_Convex" + string_end);
	}
}