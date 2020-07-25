#pragma once

#include"datastructures/DeviceData.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

namespace datastructures
{

	class HostDeviceObj
	{
	public:
		HostDeviceObj() = default;

		void to_host(void* const& dst) const
		{
			//can throw error have to add CudaError controll
			memcpy_to_host(dst);
		}

	protected:
		virtual void allocate_gpu() = 0;
		virtual void memcpy_to_device(void* const& src) = 0;
		virtual void memcpy_to_host(void* dst) const = 0;

		cudaError_t checkCuda(cudaError_t result) const
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
                