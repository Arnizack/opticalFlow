#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <memory>
#include <iomanip>
#include <fstream>

#include "optflow_composition/ContainerOptions.h"

#include "Parser/CPUBackendParser/CPUBackendParser.h"
#include "Parser/FlowContainerParser.h"

namespace json_settings
{
	namespace modern_json = nlohmann;
	
	void JsonSetupSettings(std::string json_input_file_path, std::shared_ptr<optflow_composition::ContainerOptions> options);

	modern_json::json GenerateJSON();
}