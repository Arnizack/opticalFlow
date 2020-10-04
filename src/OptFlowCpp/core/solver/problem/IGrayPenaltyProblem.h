#pragma once
#include"IGray2FrameProblem.h"
#include"../../penalty/IPenalty.h"
#include"IPenaltyProblem.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IGrayPenaltyProblem : public IGray2FrameProblem, public IPenaltyProblem
			{
			};
		}
	}
}