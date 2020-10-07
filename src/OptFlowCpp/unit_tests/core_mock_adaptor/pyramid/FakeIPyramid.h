#pragma once
#include"core/pyramid/IPyramid.h"
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"

namespace core
{
	namespace testing
	{
		using PtrProblemTyp = std::shared_ptr<IGrayPenaltyCrossProblem>;
		class FakeIPyramid : public IPyramid<PtrProblemTyp>
		{
		public:
			FakeIPyramid(int level_counter, PtrProblemTyp problem);
			


			// Inherited via IPyramid
			virtual PtrProblemTyp NextLevel() override;

			virtual bool IsEndLevel() override;
		private:
			int _counter;
			int _level_count;
			PtrProblemTyp _problem;
		};
	}
}