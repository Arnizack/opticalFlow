#pragma once

#include"CharbonnierPenalty.h"
#include<math.h>

namespace cpu_backend
{
	inline double MixValues(double val1, double val2, double mix_factor)
	{
		return val1 * (1 - mix_factor) + val2 * mix_factor;
	}
	CharbonnierPenalty::CharbonnierPenalty(std::shared_ptr<CharbonnierPenaltySettings> settings)
		: _exponent(settings->Exponent), _blend_factor(settings->DefaultBlendFactor),
		_epsilon_squared(settings->Epsilon* settings->Epsilon)
	{
	}
	double CharbonnierPenalty::ValueAt(const double& x)
	{
		
		double charbonnier = pow((x * x + _epsilon_squared) , _exponent);
		double quadratic = x * x;
		return MixValues(quadratic, charbonnier, _blend_factor);
	}
	double CharbonnierPenalty::FirstDerivativeAt(const double& x)
	{
		double charbonnier = 2 * _exponent * x * pow((x * x + _epsilon_squared), _exponent - 1);
		double quadratic = 2 * x;
		return MixValues(quadratic, charbonnier, _blend_factor);
	}
	double CharbonnierPenalty::SecondDerivativeAt(const double& x)
	{
		double charbonnier = 2 * _exponent * pow((x * x + _epsilon_squared), _exponent - 1);
		double quadratic = 2;
		return MixValues(quadratic, charbonnier, _blend_factor);
	}
	void CharbonnierPenalty::SetBlendFactor(double blend_factor)
	{
		_blend_factor = blend_factor;
	}
}