#include "logger.hpp"
#include "iLogger.hpp"
#include "consoleLogger.hpp"
#include "FileLogger.hpp"
#include <stdarg.h>
#include<vector>
#include<mutex>


namespace logger
{
	/*
	static std::vector<iLogger*> loggerInstances;
	static std::vector<logger::loggerTypes> loggerInstancesTypes;
	*/

	
	using loggerInstancesTyp = std::vector<std::pair<std::unique_ptr<iLogger>, loggerTypes>>;

	static loggerInstancesTyp loggerInstances;

	static std::mutex loggerTypesMux;

	template<typename T>
	inline decltype(auto) createPair(loggerTypes type,int level,std::string strArgs)
	{
		
		return std::pair<std::unique_ptr<iLogger>, loggerTypes>(std::make_unique<T>(level, strArgs),type);
	}

	void registerLoggerType(loggerTypes type, int level, std::string strArgs)
	{
		loggerTypesMux.lock();
		bool alreadyAdded = false;
		for (auto& loggerPair : loggerInstances)
		{
			if (type == loggerPair.second)
			{
				switch (type)
				{
				case logger::loggerTypes::FILELOGGER:
					loggerPair.first.reset(new FileLogger(level, strArgs));
					break;
				default:
					loggerPair.first.reset(new ConsoleLogger(level,strArgs));
					break;
				}
				alreadyAdded = true;
				break;
						
			}
		}
	
		if(! alreadyAdded)
		{
			switch (type)
			{
			case logger::loggerTypes::FILELOGGER:
			
				loggerInstances.push_back(createPair<FileLogger>(type, level, strArgs));
				break;
			default:

				loggerInstances.push_back(createPair<ConsoleLogger>(type, level, strArgs));
				break;
			}
		}
		loggerTypesMux.unlock();
	
	}

	void logger::log(int level, std::string msg, ...)
	{
		loggerTypesMux.lock();
		loggerTypesMux.unlock();
		for each (auto& loggerPair in loggerInstances)
		{
			auto& loggerInstance = loggerPair.first;
	
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

}