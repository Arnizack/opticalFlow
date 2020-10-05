#pragma once
#include"IGray2FrameProblem.h"
#include"../../penalty/IPenalty.h"
#include"IPenaltyProblem.h"

namespace core
{

	class IGrayPenaltyProblem : public IGray2FrameProblem, public IPenaltyProblem
	{
	};

}