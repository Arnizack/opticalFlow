#include"FakeIPyramid.h"

namespace core
{
	namespace testing
	{

		FakeIPyramid::FakeIPyramid(int level_count, std::shared_ptr<IGrayPenaltyCrossProblem> problem)
			: _counter(0), _problem(problem), _level_count(level_count)
		{
		}
	
		std::shared_ptr<IGrayPenaltyCrossProblem> FakeIPyramid::NextLevel()
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