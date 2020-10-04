#pragma once
#pragma once
#include<memory>
#include"../../IArray.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IColorCrossFilterProblem
			{
			public:
				std::shared_ptr<IArray<float, 3>> CrossFilterImage;
			};
		}
	}
}