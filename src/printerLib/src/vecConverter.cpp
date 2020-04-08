#include"vecConverter.hpp"

namespace printerLib {

	arrayWithSize convertVec(std::vector<std::string>* vec)
	{
		arrayWithSize result;
		result.feld = &(*vec)[0];
		result.size = (int)vec->size();
		std::string* feld = vec->data();
		result.feld = feld;

		return result;
	}


}