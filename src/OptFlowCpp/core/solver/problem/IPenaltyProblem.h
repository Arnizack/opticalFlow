#pragma once
#include<memory>
#include"../../penalty/IPenalty.h"
#include"../../IArray.h"

namespace core
{

	class IPenaltyProblem
	{
	public:
		std::shared_ptr < IPenalty<double>>
			PenaltyFunc{ nullptr };
	};

}