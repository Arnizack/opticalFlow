#pragma once

namespace optflow_solvers
{
	struct CGSolverSettings
	{
		double Tolerance = 1e-3;
		size_t Iterations = 100;
	};
}