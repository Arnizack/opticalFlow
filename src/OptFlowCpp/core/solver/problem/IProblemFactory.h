#pragma once
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"

namespace core
{

	class IProblemFactory
	{
	public:
		virtual std::shared_ptr<IGrayPenaltyCrossProblem> CreateGrayPenaltyCrossProblem() = 0;

	};
}