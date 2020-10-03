#pragma once
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			namespace csp = core::solver::problem;
			class IProblemFactory
			{
			public:
				virtual std::shared_ptr<csp::IGrayPenaltyCrossProblem> CreateGrayPenaltyCrossProblem() = 0;
			};
		}
	}
}