#pragma once
#include"IGrayCrossFilterProblem.h"
#include"IGrayPenaltyProblem.h"
namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IGrayPenaltyCrossProblem : public IGrayCrossFilterProblem, public IGrayPenaltyProblem
			{};
		}
	}
}