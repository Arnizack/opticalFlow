#include"printerLib.hpp"
#include<stdio.h>

namespace printerLib {

	void printSingle(std::string msg)
	{

		printf("%s\n", msg.data());
	}

}