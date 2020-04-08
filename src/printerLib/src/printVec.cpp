#include"printerLib.hpp"
#include"vecConverter.hpp"

namespace printerLib {

	void printVector(std::vector<std::string> vec)
	{
		arrayWithSize feldSize = convertVec(&vec);

		for (int i = 0; i < feldSize.size; i++)
		{
			std::string msg = feldSize.feld[i];
			printSingle(msg);
		}
	}

}