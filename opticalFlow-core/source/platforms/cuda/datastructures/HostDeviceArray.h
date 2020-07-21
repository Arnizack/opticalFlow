#pragma once

#include "../../../datastructurs/DeviceData.h"
#include "ThreadDeviceArray.h"
#include "HostDeviceObj.h"

namespace datastructures
{
	template<typename T>
	class HostDeviceArray : virtual datastructures::IDeviceArray<T>, HostDeviceObj
	{
	public:
		HostDeviceArray(T* const& arr, const size_t& ItemCount)
			: IDeviceArray<T>::IDeviceArray(arr, ItemCount)
		{}

		~HostDeviceArray()
		{
			//cudaFree(device_array.array);
		}

		constexpr size_t size() noexcept
		{
			return ItemCount;
		}

		datastructures::ThreadDeviceArray<T> getCudaArray()
		{
			//returns the gpu array struct
			//this->to_gpu();
			return device_array;
		}

	protected:
		virtual void allocate_gpu() override
		{
			//checks for cuda Errors
			checkCuda(
				cudaMalloc((void**)device_array.array, ItemCount * sizeof(T))
			);
		}

		virtual void memcpy_to_device() override
		{
			//checks for cuda Errors
			checkCuda(
				cudaMemcpy((void**)device_array.array, host_array, ItemCount * sizeof(T), cudaMemcpyHostToDevice)
			);
		}

		virtual void memcpy_to_host() override
		{
			//checks for cuda Errors
			checkCuda(
				cudaMemcpy((void**)host_array, device_array.array, ItemCount * sizeof(T), cudaMemcpyDeviceToHost)
			);
		}

	private:
		const datastructures::ThreadDeviceArray<T> device_array;
	
	};

}
                