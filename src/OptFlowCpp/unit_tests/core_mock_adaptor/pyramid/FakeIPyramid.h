#pragma once
#include"core/pyramid/IPyramid.h"
#include"core/solver/problem/IGrayPenaltyCrossProblem.h"

namespace core
{
	namespace testing
	{
		
		class FakeIPyramid : public IPyramid<IGrayPenaltyCrossProblem>
		{
		public:
			FakeIPyramid(int level_counter, std::shared_ptr<IGrayPenaltyCrossProblem> problem);
			


			// Inherited via IPyramid
			virtual std::shared_ptr<IGrayPenaltyCrossProblem> NextLevel() override;

			virtual bool IsEndLevel() override;
		private:
			int _counter;
			int _level_count;
			std::shared_ptr<IGrayPenaltyCrossProblem> _problem;
		};
	}
}