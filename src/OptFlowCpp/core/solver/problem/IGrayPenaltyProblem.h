#pragma once
#include"IGray2FrameProblem.h"
#include"../../IPenalty.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IGrayPenaltyProblem : public IGray2FrameProblem
			{
			public:
				std::shared_ptr < penalty::IPenalty<IArray<float, 2>>> PenaltyFunc;
			};
		}
	}
}