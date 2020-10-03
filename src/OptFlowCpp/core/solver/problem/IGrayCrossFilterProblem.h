#pragma once
#include"IGray2FrameProblem.h"
#include"IColorCrossFilterProblem.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IGrayCrossFilterProblem : public IGray2FrameProblem, public IColorCrossFilterProblem
			{

			};
		}
	}
}