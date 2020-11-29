#pragma once

#include"QuadraticPenalty.h"

namespace cpu_backend
{
	double QuadraticPenalty::ValueAt(const double& x)
	{
		return x * x;
	}
	double QuadraticPenalty::FirstDerivativeAt(const double& x)
	{
		return 2.0 * x;
	}
	double QuadraticPenalty::SecondDerivativeAt(const double& x)
	{
		return 2.0;
	}
}