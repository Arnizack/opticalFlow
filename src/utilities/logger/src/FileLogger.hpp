#pragma once
#include"iLogger.hpp"

class FileLogger : public iLogger
{
	private:
		int logLevel;
		std::string filePath;
	public:
		FileLogger(int level, std::string filePath);
		// Inherited via iLogger
		void log(int level, std::string msg) override;

		
};