#include<printerLib.hpp>

int main()
{
	std::vector<std::string> vec;
	vec.push_back("Hallo");
	vec.push_back("Welt");
	printerLib::printVector(vec);

	printerLib::waitOnKeyPress();
	return 0;
}