#include<logger.hpp>
#include<fstream>
#include<strstream>

#include<gtest/gtest.h>

TEST(loggerFileLogger, HalloWorld)
{
	std::string input = "Hallo World";
	logger::registerLoggerType(logger::FILELOGGER,10,"logFileTest.txt");
	logger::log(10, input);

	std::ifstream t("logFileTest.txt");
	std::string output((std::istreambuf_iterator<char>(t)),
		std::istreambuf_iterator<char>());

	EXPECT_EQ("LEVEL 10: "+input + "\n", output);
}

TEST(loggerFileLogger, HalloWorldButToLow)
{
	std::string input = "Hallo World";
	logger::registerLoggerType(logger::FILELOGGER, 0, "logFileTest.txt");
	logger::registerLoggerType(logger::FILELOGGER, 10, "logFileTest.txt");
	logger::log(4, input);
	std::ifstream t("logFileTest.txt");
	std::string output((std::istreambuf_iterator<char>(t)),
		std::istreambuf_iterator<char>());
	EXPECT_EQ("", output);
	//EXPECT_TRUE("" == output);
}

TEST(loggerFileLogger, HalloWorldFormated)
{
	std::string input = "Hallo World%s%d";
	logger::registerLoggerType(logger::FILELOGGER, 10, "logFileTest.txt");
	logger::log(10, input," Die Zahl ist: ",10);
	std::ifstream t("logFileTest.txt");
	std::string output((std::istreambuf_iterator<char>(t)),
		std::istreambuf_iterator<char>());
	EXPECT_EQ("LEVEL 10: Hallo World Die Zahl ist: 10\n" , output);
}


TEST(loggerFileLogger, HalloWorldMultiLine)
{
	logger::registerLoggerType(logger::FILELOGGER, 10, "logFileTest.txt");
	logger::log(10, "Hallo");
	logger::log(22, "World");
	std::ifstream t("logFileTest.txt");
	std::string output((std::istreambuf_iterator<char>(t)),
		std::istreambuf_iterator<char>());

	EXPECT_EQ("LEVEL 10: Hallo\nLEVEL 22: World\n", output);
}