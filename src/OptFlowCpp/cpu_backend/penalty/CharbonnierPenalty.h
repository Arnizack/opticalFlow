#pragma once
#include"core/penalty/IBlendablePenalty.h"
namespace cpu_backend
{
	struct CharbonnierPenaltySettings
	{
		double DefaultBlendFactor = 0;
		double Epsilon = 0.001;
	};
	class CharbonnierPenalty : public core::IBlendablePenalty<double>
	{
	public:
		CharbonnierPenalty(CharbonnierPenaltySettings settings);
		// Inherited via IBlendablePenalty
		virtual double ValueAt(const double& x) override;
		virtual double FirstDerivativeAt(const double& x) override;
		virtual double SecondDerivativeAt(const double& x) override;
		virtual void SetBlendFactor(double blend_factor) override;

	private:
		double _blend_factor;
		double _epsilon_squared;
	};
}