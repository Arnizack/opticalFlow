#pragma once
#include <fstream>

#include <nlohmann/json.hpp>
#include <memory>
#include <iomanip>
#include <fstream>

#include "optflow_composition/ContainerOptions.h"
#include "optflow_solvers/solvers/settings/GNCPenaltySolverSettings.h"

#include "Parser/CPUBackendParser/CPUBackendParser.h"
#include "Parser/FlowSolverParser/CGSolverParser.h"
#include "Parser/FlowSolverParser/GNCPenaltySolverParser.h"
#include "Parser/FlowSolverParser/IncrementalSolverParser.h"
#include "Parser/FlowSolverParser/LinearizationSolverParser.h"
#include "Parser/FlowContainerParser.h"
#include "Parser/LevelContainerParser.h"
#include "Parser/PyramidContainerParser.h"

namespace console_ui
{
	namespace modern_json = nlohmann;

	std::shared_ptr<optflow_composition::ContainerOptions> JsonSetupSettings (std::string json_input_file_path, std::shared_ptr<optflow_composition::ContainerOptions> options);

	modern_json::json GenerateJSON();

	void OutputJSON(const modern_json::json& output, const std::string& file_path);
}