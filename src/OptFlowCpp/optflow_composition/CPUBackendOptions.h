#pragma once
#include "cpu_backend/penalty/CharbonnierPenalty.h"
#include "cpu_backend/sb_linearsystem/SunBakerLSUpdater.h"
#include "cpu_backend/flow/CrossBilateralMedianFilter.h"

#include <memory>

namespace optflow_composition
{
	struct CPUBackendOptions
	{
		std::shared_ptr<cpu_backend::CrossMedianFilterSettings> CrossMedianFilter = std::make_shared<cpu_backend::CrossMedianFilterSettings>(cpu_backend::CrossMedianFilterSettings());
		std::shared_ptr<cpu_backend::LinearSystemSettings> LinearSystem = std::make_shared<cpu_backend::LinearSystemSettings>(cpu_backend::LinearSystemSettings());
		std::shared_ptr<cpu_backend::CharbonnierPenaltySettings> CharbonnierPenalty = std::make_shared<cpu_backend::CharbonnierPenaltySettings>(cpu_backend::CharbonnierPenaltySettings());
	};
}