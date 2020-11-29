#pragma once
#include"IGrayCrossFilterProblem.h"
#include"IGrayPenaltyProblem.h"
namespace core
{

	class IGrayPenaltyCrossProblem : public IGray2FrameProblem,
		public IColorCrossFilterProblem, public IPenaltyProblem
	{};

}