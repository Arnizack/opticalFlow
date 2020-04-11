#include<logger.hpp>


#include<gtest/gtest.h>

TEST(loggerConsoleLogger, HalloWorld)
{
	std::string input = "Hallo World";
	testing::internal::CaptureStdout();
	logger::registerLoggerType(logger::CONSOLELOGGER,10,"");
	logger::log(10, input);
	std::string output = testing::internal::GetCapturedStdout();
	EXPECT_EQ(input + "\n", output);
}

TEST(loggerConsoleLogger, HalloWorldButToLow)
{
	std::string input = "Hallo World";
	testing::internal::CaptureStdout();
	logger::registerLoggerType(logger::CONSOLELOGGER, 0, "");
	logger::registerLoggerType(logger::CONSOLELOGGER, 10, "");
	logger::log(4, input);
	std::string output = testing::internal::GetCapturedStdout();
	EXPECT_EQ("", output);
	//EXPECT_TRUE("" == output);
}

TEST(loggerConsoleLogger, HalloWorldFormated)
{
	std::string input = "Hallo World%s%d";
	testing::internal::CaptureStdout();
	logger::registerLoggerType(logger::CONSOLELOGGER, 10, "");
	logger::log(10, input," Die Zahl ist: ",10);
	std::string output = testing::internal::GetCapturedStdout();
	EXPECT_EQ("Hallo World Die Zahl ist: 10\n" , output);
}