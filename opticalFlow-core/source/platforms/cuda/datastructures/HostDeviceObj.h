#pragma once

#include "../../../datastructurs/DeviceData.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

namespace datastructures
{

	class HostDeviceObj : datastructures::IDeviceObj
	{
	public:
		HostDeviceObj() = default;
		
		virtual void fetchData()
		{
			//can throw error have to add CudaError controll
			memcpy_to_host();
		}

	protected:
		virtual void allocate_gpu() = 0;
		virtual void memcpy_to_device() = 0;
		virtual void memcpy_to_host() = 0;
		virtual void to_device()
		{
			//can throw error have to add CudaError controll
			allocate_gpu();
			memcpy_to_device();
		}

		cudaError_t checkCuda(cudaError_t result)
		{
			if (result != cudaSuccess)
			{
				fprintf(stderr, "CUDA Runtime Error %s\n", cudaGetErrorString(result));	//writes to Error stream
				assert(((void)"CUDA Runtime Error: ", result != cudaSuccess));
			}

			return result;
		}


	};

}
                