#pragma once

#include<string>

class iLogger
{
	public:
		virtual void log(int level, std::string msg) = 0;
};