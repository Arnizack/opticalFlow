#pragma once
#include"core/penalty/IPenalty.h"
namespace cpu_backend
{
	class QuadraticPenalty : public core::IPenalty<double>
	{
		// Inherited via IPenalty
		virtual double ValueAt(const double& x) override;
		virtual double FirstDerivativeAt(const double& x) override;
		virtual double SecondDerivativeAt(const double& x) override;
	};
}