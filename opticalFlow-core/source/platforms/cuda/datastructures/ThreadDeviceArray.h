#pragma once

#include <cuda_runtime.h>

namespace datastructures
{
	template<typename T>
	struct ThreadDeviceArray
	{
		T* array = 0;

		ThreadDeviceArray() = default;
		ThreadDeviceArray(const ThreadDeviceArray& obj) = delete;

		~ThreadDeviceArray()
		{
			cudaFree(array);
		}

		//Device Operators
		__device__
			T& operator[](const int& i)
		{
			array[i];
		}

		__device__
			const T& operator[](const int& i) const
		{
			array[i];
		}
	};

}
