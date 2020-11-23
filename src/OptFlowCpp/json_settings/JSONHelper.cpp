#pragma once
#include"JSONHelper.h"

namespace json_settings
{
	inline void RepalaceAll(std::string& in, const std::string& target_1, const std::string& target_2, const std::string& replace_with = "_")
	{
		size_t pos = 0;

		while ((pos = in.find(target_1, pos)) != std::string::npos) {
			in.replace(pos, target_1.length(), replace_with);
			pos += replace_with.length();
		}

		pos = 0;

		while ((pos = in.find(target_2, pos)) != std::string::npos) {
			in.replace(pos, target_2.length(), replace_with);
			pos += replace_with.length();
		}
	}

	inline std::string all_lower(std::string in)
	{
		std::transform(in.begin(), in.end(), in.begin(), [](unsigned char c) { return std::tolower(c); });

		return in;
	}

	std::string GetOptionName(std::string var_name, const std::string& suffix)
	{
		RepalaceAll(var_name, ".", "->", "_");

		if (suffix.empty() == true)
			return all_lower(var_name);

		return all_lower(var_name + '_' + suffix);
	}

	void OutputJSON(const modern_json::json& output, const std::string& file_path)
	{
		std::ofstream out_stream(file_path);

		out_stream << std::setw(output.size() * 2) << output << std::endl;
	}
}