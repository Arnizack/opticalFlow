#pragma once
#include"IGray2FrameProblem.h"
#include"IColorCrossFilterProblem.h"

namespace core
{

	class IGrayCrossFilterProblem : public IGray2FrameProblem, public IColorCrossFilterProblem
	{

	};

}