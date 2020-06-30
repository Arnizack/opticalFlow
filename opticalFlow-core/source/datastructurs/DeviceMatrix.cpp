#include "DeviceMatrix.h"

namespace datastructures
{
	template<typename T, size_t VectorDimensions>
	datastructures::Device2DMatrix<T, VectorDimensions>::Device2DMatrix(const size_t& count, const T *const data)
		: Heigth(VectorDimensions), Width(count), size(Heigth * Width), host_data(data)
	{
		//Allocate space on the global Memory of the GPU
		checkCuda(cudaMalloc((void **)&this->device_data, this->size));

		//copy data from CPU to GPU
		checkCuda(cudaMemcpy(this->device_data, this->host_data, this->size, cudaMemcpyHostToDevice));
	}

	template<typename T, size_t VectorDimensions>
	Device2DMatrix<T, VectorDimensions>::~Device2DMatrix()
	{
		//copy back data from GPU to CPU
		checkCuda(cudaMemcpy(device_data, host_data, size, cudaMemcpyDeviceToHost));

		//speicher wieder freigeben
		checkCuda(cudaFree(device_data));
	}

	template<typename T, size_t VectorDimensions>
	__device__ T & Device2DMatrix<T, VectorDimensions>::operator[](const int2 & i)
	{
		return this->device_data[i];
	}

	template<typename T, size_t VectorDimensions>
	__device__ 
		const T & Device2DMatrix<T, VectorDimensions>::operator[](const int2 & i) const
	{
		return this->device_data[i];
	}

	template<typename T, size_t VectorDimensions>
	datastructures::Device2DMatrix<T, VectorDimensions>::checkCuda(cudaError_t result)
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
