#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <memory>

#include "optflow_composition/ContainerOptions.h"
#include "optflow_solvers/solvers/settings/GNCPenaltySolverSettings.h"

namespace console_ui
{
	namespace modern_json = nlohmann;

	std::shared_ptr<optflow_composition::ContainerOptions> JsonSetupSettings (std::string json_input_file_path, std::shared_ptr<optflow_composition::ContainerOptions> options);

}