#pragma once
#include<memory>
#include"../../penalty/IPenalty.h"
#include"../../IArray.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IPenaltyProblem
			{
			public:
				std::shared_ptr < penalty::IPenalty<IArray<float, 2>>> PenaltyFunc;
			};
		}
	}
}