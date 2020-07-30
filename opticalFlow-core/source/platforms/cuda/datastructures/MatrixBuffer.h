#pragma once
#include"kernelInfo.cuh"

#include<cuda_runtime.h>
#include<cuda_device_runtime_api.h>

namespace cuda
{
	template<typename T>
	struct MatrixBufferRF
	{
	private:
		T* Data;
		int Width;
	
	public:
		__device__ MatrixBufferRF(T* data,const int& width):
			Data(data), Width(width)
		{}
		__device__ T& operator[](const int2& idx)
		{
			return Data[idx.x + idx.y * Width];
		}

	};

	template<class T>
	MatrixBufferRF<T> allocMatrixBufferRF(KernelInfo& kinfo, const int& width, const int& heigth)
	{
		auto buffer = MatrixBufferRF<T>(kinfo.SharedMemStart, width);
		kinfo.SharedMemStart += width * heigth * sizeof(T);
		return buffer;
	}
}