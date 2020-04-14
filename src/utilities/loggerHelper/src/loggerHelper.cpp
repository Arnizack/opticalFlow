#include"loggerHelper.hpp"
#include<stdarg.h>
#include<sstream>

void logger::logLine(int logLevel, char zeichen, int length)
{
	std::string line(length,zeichen);

	logger::log(logLevel, line);

}

void logger::logLine(int logLevel, char zeichen)
{
	logger::logLine(logLevel, zeichen, 100);
}

void logger::logLine(int logLevel, int length)
{
	logger::logLine(logLevel, '-', length);
}

void logger::logLine(int logLevel)
{
	logger::logLine(logLevel, '-', 100);
}

void logger::logLine()
{
	logger::logLine(DEBUG_LEVEL, '-', 100);
}

template<typename T>
std::string convertToString(T& obj)
{
	std::ostringstream ss;
	ss << obj;
	return ss.str();

}


void logger::logFunctionBegin(int level, std::string functionName)
{
	logLine(level);
	logger::log(level, functionName);
	logLine(level);
}


void logger::logFunctionBegin(std::string functionName)
{
	logFunctionBegin(DEBUG_LEVEL, functionName);
}
