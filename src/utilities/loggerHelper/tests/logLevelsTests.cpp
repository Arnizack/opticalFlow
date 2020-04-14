#include<gtest/gtest.h>
#include<loggerHelper.hpp>

TEST(LoggerHelper, logLine)
{

	logger::registerLoggerType(logger::loggerTypes::CONSOLELOGGER, 10, "");
	
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
	logger::registerLoggerType(logger::loggerTypes::CONSOLELOGGER, 10, "");

	testing::internal::CaptureStdout(); 

	logger::logArgsDefault("int", "double", "komisch");

	std::vector<int> vec = { 1, 2, 3, 4 };

	logger::logArgsDefault(1, 2.0, vec);

	std::string output = testing::internal::GetCapturedStdout();

	std::string expected = "int                 |double              |komisch             |\n1                   |2                   |[1, 2, 3, 4]        |\n";

	EXPECT_EQ(expected, output);

}

TEST(LoggerHelper, log2DData)
{

	logger::registerLoggerType(logger::loggerTypes::CONSOLELOGGER, 10, "");

	testing::internal::CaptureStdout();
	
	std::vector<std::vector<int>> vec = { {1,2},{3,45} };

	logger::log2DData(10, vec);
	std::string output = testing::internal::GetCapturedStdout();
	EXPECT_EQ("1  2  \n3  45 \n", output);
}

TEST(LoggerHelper, logFunctionBegin)
{

	logger::registerLoggerType(logger::loggerTypes::CONSOLELOGGER, 10, "");

	testing::internal::CaptureStdout();

	logger::logFunctionBegin("Test Func");

	std::string output = testing::internal::GetCapturedStdout();
	std::string expected = "----------------------------------------------------------------------------------------------------\nTest Func\n----------------------------------------------------------------------------------------------------\n";
	EXPECT_EQ(expected, output);
}


TEST(LoggerHelper, logFunctionEnd)
{

	logger::registerLoggerType(logger::loggerTypes::CONSOLELOGGER, 10, "");

	testing::internal::CaptureStdout();

	logger::logFunctionEnd("Test Func",10);

	std::string output = testing::internal::GetCapturedStdout();
	EXPECT_EQ("End of Test Func: 10\n", output);
}