#pragma once

#include<vector>

namespace printerLib {

	struct arrayWithSize
	{
		std::string* feld;
		int size;
	};

	arrayWithSize convertVec(std::vector<std::string>* vec);


}