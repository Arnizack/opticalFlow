#pragma once
#include "DeviceArray.h"

namespace datastructures
{
	datastructures::DeviceArray<T>::DeviceArray(const size_t& count, T *const data)
		: ItemCount(count), size(count * sizeof(T)), host_data(data)
	{
		//Allocate space on the global Memory of the GPU
		checkCuda(cudaMalloc((void **)&this->device_data, this->size));

		//copy data from CPU to GPU
		checkCuda(cudaMemcpy(this->device_data, this->host_data, this->size, cudaMemcpyHostToDevice));
	}

	template<typename T>
	datastructures::DeviceArray<T>::~DeviceArray()
	{
		//copy back data from GPU to CPU
		checkCuda(cudaMemcpy(device_data, host_data, size, cudaMemcpyDeviceToHost));

		//speicher wieder freigeben
		checkCuda(cudaFree(device_data));
	}

	template<typename T>
	__device__
		const T& datastructures::DeviceArray<T>::operator[](const int& i) const
	{
		return this->device_data[i];
	}

	template<typename T>
	__device__
		T& datastructures::DeviceArray<T>::operator[](const int& i)
	{
		return this->device_data[i];
	}

	template<typename T>
	cudaError_t datastructures::DeviceArray<T>::checkCuda(cudaError_t result)
	{
		/*
			Checks Cuda Methods if they work normal
			does not do anything if in release
		*/

#if defined (DEBUG) || (_DEBUG)
		if (result != cudaSuccess)
		{
			fprintf(stderr, "CUDA Runtime Error %s\n", cudaGetErrorString(result));	//writes to Error stream
			assert(((void) "CUDA Runtime Error: ", result != cudaSuccess));
		}
#endif
		return result;
	}
}
