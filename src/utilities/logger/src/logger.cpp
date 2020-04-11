#include "logger.hpp"
#include "iLogger.hpp"
#include "consoleLogger.hpp"
#include "FileLogger.hpp"
#include <stdarg.h>
#include<vector>

static std::vector<iLogger*> loggerInstances;
static std::vector<logger::loggerTypes> loggerInstancesTypes;


void logger::registerLoggerType(loggerTypes type, int level, std::string strArgs)
{
	iLogger* ptrloggerInst = 0;
	switch (type)
	{
	case FILELOGGER:
	{
		
		FileLogger temp(level, strArgs);
		static FileLogger fLogger = temp;
		ptrloggerInst = &(fLogger);
	}
		break;
	

	default:
		
		static ConsoleLogger cLogger(level, strArgs);

		ptrloggerInst = &(cLogger);

		

	}

	if (std::count(loggerInstancesTypes.begin(), loggerInstancesTypes.end(), type) == 0)
	{
		loggerInstances.push_back(ptrloggerInst);
		loggerInstancesTypes.push_back(type);
	}
	
}

void logger::log(int level, std::string msg, ...)
{
	for each (iLogger* loggerInstance in loggerInstances)
	{

	
		if (loggerInstance == 0)
		{
			printf("logger Instance is not initialized\n");

		}
		else
		{
			char dest[1024 * 16];
			va_list argptr;
			va_start(argptr, msg);
		
			vsprintf_s(dest, msg.data(), argptr);
			va_end(argptr);

			loggerInstance->log(level, dest);
		}
	}
}

