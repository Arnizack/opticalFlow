#include"FakeIPyramid.h"

namespace core
{
	namespace testing
	{
		using PtrProblemTyp = std::shared_ptr<IGrayPenaltyCrossProblem>;
		FakeIPyramid::FakeIPyramid(int level_count, PtrProblemTyp problem)
			: _counter(0), _problem(problem), _level_count(level_count)
		{
		}
		PtrProblemTyp FakeIPyramid::NextLevel()
		{
			_counter++;
			return _problem;
		}
		bool FakeIPyramid::IsEndLevel()
		{
			return _counter >= _level_count;
		}
		
	}
}