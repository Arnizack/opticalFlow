#pragma once
#include<chrono>
#include<string>

namespace test_utilities
{
	class Timer
	{
	public:
		Timer(std::string name);

		void Stop();
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
		std::string Name;
	};
}