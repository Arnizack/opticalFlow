#include<gtest/gtest.h>
#include<vecConverter.hpp>


TEST(printerLibTESTS, vecConverterTest)
{
	std::vector<std::string> vec;
	vec.push_back("Hallo");
	vec.push_back("Wie");
	vec.push_back("gehts");
	printerLib::arrayWithSize felds = printerLib::convertVec(&vec);

	for (int i = 0; i < felds.size; i++)
	{
		EXPECT_TRUE(vec[i]== felds.feld[i]);
	}
}