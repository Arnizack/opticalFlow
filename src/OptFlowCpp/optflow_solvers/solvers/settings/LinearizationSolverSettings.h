#pragma once

namespace optflow_solvers
{
	struct LinearizationSolverSettings
	{
		double StartRelaxation = 1e-04 / 255.0;
		double EndRelaxation = 1e-01 / 255.0;
		double RelaxationSteps = 3;
	};
}