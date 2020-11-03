#pragma once
#include "core/solver/problem/IProblemFactory.h"

namespace cpu_backend
{
	class ProblemFactory : public core::IProblemFactory
	{
	public:
		virtual std::shared_ptr<core::IGrayPenaltyCrossProblem> CreateGrayPenaltyCrossProblem() override
		{
			return std::make_shared<core::IGrayPenaltyCrossProblem>( core::IGrayPenaltyCrossProblem() );
		}
	};
}