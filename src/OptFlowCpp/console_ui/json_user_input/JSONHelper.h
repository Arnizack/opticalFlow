#pragma once
#include <string>
#include <nlohmann/json.hpp>

#ifndef VarName(var) (#var)
#define VarName(var) (#var)
#endif // !VarName


namespace console_ui
{
	namespace modern_json = nlohmann;

	/*
	* prefix_(var_name)_suffix
	* prefix = until first .
	* suffix = starting at last .
	*/
	std::string GetOptionName(std::string var_name, const std::string& suffix = "");
}