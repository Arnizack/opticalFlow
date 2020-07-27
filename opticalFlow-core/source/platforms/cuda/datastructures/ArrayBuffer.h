#pragma once
#include"kernelInfo.cuh"

#include<cuda_runtime.h>
#include<cuda_device_runtime_api.h>

namespace cuda
{
	template<typename T>
	T* allocArrayBufferRF(KernelInfo& kinfo,const int& count)
	{
		T* array = static_cast<T*>(kinfo.SharedMemStart);
		kinfo.SharedMemStart += count * sizeof(T);
		return array;
	}
}
