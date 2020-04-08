#include<gtest/gtest.h>
#include<sample.hpp>


TEST(printerLibTESTS, vecConverterTest)
{
	EXPECT_TRUE("Hallo World"==demoLib::getHalloWorld());
}