#include<chrono>
namespace core
{
class Timer
{
public:
	Timer(int logLevel = 10)
	{
		m_StartTimePoint = std::chrono::high_resolution_clock::now();
		LogLevel = logLevel;
	}

	void Stop()
	{
		auto endTimepoint = std::chrono::high_resolution_clock::now();
		auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_StartTimePoint).time_since_epoch().count();
		auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endTimepoint).time_since_epoch().count();
		auto duration = end - start;
		double ms = duration * 0.001;

		logger::log(LogLevel, "Duration: %lf us", (double)duration);
		logger::log(LogLevel, "Duration: %lf ms", ms);
	}
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimePoint;
	int LogLevel;
};

}