#pragma once
#include"../../IArray.h"
#include<memory>
namespace core
{
	namespace solver
	{
		namespace problem
		{
			class IColor2FrameProblem
			{
			public:
				std::shared_ptr<core::IArray<float, 3>> first_frame{ nullptr };
				std::shared_ptr<core::IArray<float, 3>> second_frame{nullptr};

			};
		}
	}
}