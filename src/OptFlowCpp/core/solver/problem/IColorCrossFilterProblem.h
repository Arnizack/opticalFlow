#pragma once
#pragma once
#include<memory>
#include"../../IArray.h"

namespace core
{

	class IColorCrossFilterProblem
	{
	public:
		std::shared_ptr<IArray<float, 3>> CrossFilterImage{ nullptr };
	};

}