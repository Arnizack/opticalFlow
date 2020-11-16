#pragma once
#include"Hypodermic/Container.h"
#include"optflow_solvers/solvers/settings/CGSolverSettings.h"
#include"optflow_solvers/solvers/settings/GNCPenaltySolverSettings.h"
#include"optflow_solvers/solvers/settings/IncrementalSolverSettings.h"
#include"optflow_solvers/solvers/settings/LinearizationSolverSettings.h"


namespace console_ui
{
	struct Backends
	{
		std::shared_ptr<Hypodermic::Container> CPUBackend;
		Backends(std::shared_ptr<Hypodermic::Container> cpu_backend)
			: CPUBackend(cpu_backend)
		{}

	};

	struct LevelSettings
	{
		std::shared_ptr<optflow_solvers::LinearizationSolverSettings> Linearization = std::make_shared< optflow_solvers::LinearizationSolverSettings>();
		std::shared_ptr<optflow_solvers::IncrementalSolverSettings> Incremental = std::make_shared<optflow_solvers::IncrementalSolverSettings>();
		std::shared_ptr<optflow_solvers::CGSolverSettings> CGsolver = std::make_shared<optflow_solvers::CGSolverSettings>();
	};

	struct PyramidResolution
	{
		double ScaleFactor = 0.5;
		size_t MinResolutionX = 32;
		size_t MinResolutionY = 32;
	};

	struct PyramidSettings
	{
		LevelSettings SingleLevelSettings;
		PyramidResolution Resolution;


	};

	class PyramidIteratorSettings
	{
	public:
		PyramidSettings ConvexSettings;
		PyramidSettings NonConvexSettings;

	};

	struct GNCSolverSettings
	{
		std::shared_ptr<optflow_solvers::GNCPenaltySolverSettings> GNCSettings = std::make_shared<optflow_solvers::GNCPenaltySolverSettings>();
		PyramidIteratorSettings PyramidContainerSettings;
	};
}