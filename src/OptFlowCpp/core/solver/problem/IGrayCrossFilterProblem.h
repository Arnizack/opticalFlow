#pragma once
#include"IGray2FrameProblem.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IGrayCrossFilterProblem : public IGray2FrameProblem
			{
			public:
				std::shared_ptr<IArray<float, 3>> CrossFilterImage;
			};
		}
	}
}