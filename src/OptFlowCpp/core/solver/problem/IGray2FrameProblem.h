#pragma once
#include<memory>
#include"../../IArray.h"

namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IGray2FrameProblem
			{
			public:
				std::shared_ptr<core::IArray<float, 2>> FirstFrame{nullptr};
				std::shared_ptr<core::IArray<float, 2>> SecondFrame{nullptr};
			};
		}
	}
}