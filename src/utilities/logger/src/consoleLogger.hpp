#include"iLogger.hpp"

class ConsoleLogger : public iLogger
{
	private: 
		int logLevel;

	public: 
		ConsoleLogger(int level, std::string strArgs);

		void log(int level, std::string msg) const override;

};