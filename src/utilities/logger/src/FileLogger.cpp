#include "FileLogger.hpp"
#include<fstream>
#pragma once


FileLogger::FileLogger(int level, std::string filePath)
{
	this->logLevel = level;
	this->filePath = filePath;
	std::ofstream file;
	file.open(this->filePath, std::ofstream::out);
	file.clear();
	file.close();

}

void FileLogger::log(int level, std::string msg) const
{
	if (level >= this->logLevel)
	{
		std::ofstream file;
		file.open(this->filePath, std::ios_base::app);
		file << "LEVEL ";
		file << level;
		file << ": ";
		file << msg;
		file << "\n";
		file.close();
	}
}
	
