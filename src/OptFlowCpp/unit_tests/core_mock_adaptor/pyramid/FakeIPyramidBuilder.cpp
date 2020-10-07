#pragma once
#include"FakeIPyramidBuilder.h"
#include"FakeIPyramid.h"

namespace core
{
	namespace testing
	{
		using PtrProblemTyp = std::shared_ptr<IGrayPenaltyCrossProblem>;

		std::shared_ptr<IPyramid<PtrProblemTyp>> 
			FakeIPyramidBuilder::Create(PtrProblemTyp last_level)
		{

			return _pyramid;
		}
		FakeIPyramidBuilder::FakeIPyramidBuilder(std::shared_ptr<IPyramid<PtrProblemTyp>> pyramid)
			:_pyramid(pyramid)
		{
		}
	}
}