#pragma once

#include<string>




namespace logger
{
	
	enum loggerTypes
	{
		CONSOLELOGGER,
		FILELOGGER
	};
	
	/*
	Create a new Logger if no Logger of the given type exists.
	Updates and resests the Logger if a Logger of the given typ exists
	*/
	void registerLoggerType(loggerTypes type, int level, std::string strArgs);

	/*
	logges a Message, the Message can be formatted similarly to printf
	Example:
		log(10, "Hallo World %d", 4) logges at Level 10: Hallo World 4
	*/
	void log(int level, std::string msg,...);
};

