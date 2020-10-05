#pragma once
#include<memory>
#include"../../IArray.h"

namespace core
{


	class IGray2FrameProblem
	{
	public:
		std::shared_ptr<IArray<float, 2>> FirstFrame{ nullptr };
		std::shared_ptr<IArray<float, 2>> SecondFrame{ nullptr };
	};

}