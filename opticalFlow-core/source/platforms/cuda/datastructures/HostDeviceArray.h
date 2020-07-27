#pragma once

#include "datastructures/DeviceData.h"
#include "HostDeviceObj.h"

namespace datastructures
{
	template<typename T>
	class HostDeviceArray : public datastructures::IDeviceArray<T>, public HostDeviceObj
	{
	public:
		HostDeviceArray(T* const& arr, const size_t& ItemCount)
			: IDeviceArray<T>::IDeviceArray(ItemCount)
		{
			allocate_gpu();
			memcpy_to_device(arr);
		}

		virtual ~HostDeviceArray(void) override
		{
			cudaFree(device_array);
		}

		constexpr size_t size() noexcept
		{
			return ItemCount;
		}

		T* getCudaArray()
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
				cudaMalloc(&device_array, ItemCount * sizeof(T))
			);
		}

		virtual void memcpy_to_device(void* const& src) override
		{
			//checks for cuda Errors
			checkCuda(
				cudaMemcpy(device_array, src, ItemCount * sizeof(T), cudaMemcpyHostToDevice)
			);
		}

		virtual void memcpy_to_host(void* dst) const override
		{
			//checks for cuda Errors
			checkCuda(
				cudaMemcpy(dst, device_array, ItemCount * sizeof(T), cudaMemcpyDeviceToHost)
			);
		}

	private:
		T* device_array;
	};

}
                