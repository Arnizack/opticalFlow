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
	logger::logLine(10, '-', 100);
}

template<typename T>
std::string convertToString(T& obj)
{
	std::ostringstream ss;
	ss << obj;
	return ss.str();

}

void fillStrWithMaxLength(const size_t length, const std::string& str, const size_t start, std::string* fillStr)
{
	for (size_t i = 0; i < str.size() && i < length; i++)
	{
		fillStr->operator[](i) = str[i];
	}

}


