#pragma once
#include"PyramidBuilder.h"
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"
namespace optflow_solvers
{
	class GrayPenaltyCrossPyramidBuilder : public PyramidBuilder<core::IGrayPenaltyCrossProblem>
	{
	public:
		GrayPenaltyCrossPyramidBuilder(std::shared_ptr<core::IScaler<core::IGrayPenaltyCrossProblem>> scaler);

		// Inherited via PyramidBuilder
		virtual std::shared_ptr<core::IPyramid<core::IGrayPenaltyCrossProblem>> 
			Create(std::shared_ptr<core::IGrayPenaltyCrossProblem> last_level) override;

	};
}