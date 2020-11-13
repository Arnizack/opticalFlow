#pragma once
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"
#include"core/solver/problem/IGrayCrossFilterProblem.h"
#include"core/IArray.h"

namespace core
{

	class IProblemFactory
	{
	public:
		virtual std::shared_ptr<IGrayPenaltyCrossProblem> CreateGrayPenaltyCrossProblem() = 0;
		virtual std::shared_ptr<IGrayCrossFilterProblem> CreateGrayCrossFilterProblem(std::shared_ptr<IArray<float,3>> first_image, std::shared_ptr<IArray<float,3>> sconde_image) = 0;
	};
}