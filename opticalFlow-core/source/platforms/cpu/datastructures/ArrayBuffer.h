#pragma once
#include<vector>

namespace cpu
{
	template<class T>
	std::vector<T> allocArrayBuffer(int size)
	{
		return std::vector<T>(size);
	}
}