#include<gtest/gtest.h>
#include<loggerHelper.hpp>

TEST(LoggerHelper, logLine)
{

	logger::registerLoggerType(logger::CONSOLELOGGER, 10, "");
	
	testing::internal::CaptureStdout();
	logger::logLine();
	std::string output = testing::internal::GetCapturedStdout();
	EXPECT_EQ("----------------------------------------------------------------------------------------------------\n", output);
}


void lol(std::string arg[])
{
	printf("");
}

TEST(LoggerHelper, logArgs1)
{
	logger::registerLoggerType(logger::CONSOLELOGGER, 10, "");

	testing::internal::CaptureStdout(); 

	logger::logArgsDefault("int", "double", "komisch");
	logger::logArgsDefault(1, 2.0,std::vector<int>());

	std::string output = testing::internal::GetCapturedStdout();
	EXPECT_EQ("", output);

}