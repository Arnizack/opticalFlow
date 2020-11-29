#pragma once
#include <string>
#include <nlohmann/json.hpp>
#include <iomanip>
#include <fstream>

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName


namespace json_settings
{
	namespace modern_json = nlohmann;

	/*
	* prefix_(var_name)_suffix
	* prefix = until first .
	* suffix = starting at last .
	*/
	std::string GetOptionName(std::string var_name, const std::string& suffix = "");

	void OutputJSON(const modern_json::json& output, const std::string& file_path);
}