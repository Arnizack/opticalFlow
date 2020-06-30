#pragma once
#include "DeviceArray.hpp"
#include <cuda_runtime.h>

namespace datastructures
{
	template<typename T>
	class DeviceArray 
	{
	public:
		DeviceArray(const size_t& count, T *const data = NULL);
		~DeviceArray();

		//Device and Host Operators
		__device__ 
			T& operator[](const int& i);
		__device__ 
			const T& operator[](const int& i) const;

	protected:
		size_t ItemCount;
		size_t size;
		T *host_data;
		T *device_data = nullptr;
		inline cudaError_t checkCuda(cudaError_t result);
	};
}
