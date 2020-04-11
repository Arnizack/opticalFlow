#include"ConsoleLogger.hpp"

ConsoleLogger::ConsoleLogger(int level, std::string strArgs)
{
	this->logLevel = level;
}

void ConsoleLogger::log(int level,std::string msg)
{
	if (this->logLevel <= level)
	{
		printf(msg.data());
		printf("\n");
	}
}
