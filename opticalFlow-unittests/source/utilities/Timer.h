#pragma once
#include<chrono>
#include<string>

namespace utilities
{
	class Timer
	{
	public:
		Timer( std::string name)
		{
			Name = name;
			m_StartTimePoint = std::chrono::high_resolution_clock::now();
			
		}

		void Stop()
		{
			auto endTimepoint = std::chrono::high_resolution_clock::now();
			auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_StartTimePoint).time_since_epoch().count();
			auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endTimepoint).time_since_epoch().count();
			auto duration = end - start;
			double ms = duration * 0.001;

			printf("%s: Duration: %lf us\n",Name.data(), (double)duration);
			printf( "%s: Duration: %lf ms\n",Name.data(), ms);
		}
	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
		std::string Name;
	};

}