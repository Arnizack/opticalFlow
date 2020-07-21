#pragma once

#include "../../../datastructurs/DeviceData.h"
#include "HostDeviceObj.h"
#include "ThreadDevice2DMatrix.h"

namespace datastructures
{
	template<typename T, size_t VectorDimensions>
	class HostDevice2DMatrix : virtual datastructures::IDevice2DMatrix<T, VectorDimensions>, HostDeviceObj
	{
		HostDevice2DMatrix(const T*& matrix, const size_t& width, const size_t& height)
			: IDevice2DMatrix<T>{ matrix, width, height }, device_matirx{ datastructures::ThreadDevice2DMatrix<T> temp(width) }
		{}

		~HostDevice2DMatrix()
		{
			//cudaFree(device_matirx.matrix);
		}

		constexpr size_t size() noexcept
		{
			return VectorDimensions;
		}

		datastructures::ThreadDevice2DMatrix getCuda2DMatrix()
		{
			//could return not initialized Data
			return device_matirx;
		}

	protected:
		virtual void allocate_gpu()
		{
			//allocates Memory on the gpu for the matrix
			checkCuda(
				cudaMalloc((void**)device_matrix.matrix, VectorDimensions * sizeof(T)) // vlt (void**)
			);
		}

		virtual void memcpy_to_device()
		{
			//copies the data to gpu
			checkCuda(
				cudaMemcpy((void**)device_matrix.matrix, host_array, VectorDimensions * sizeof(T), cudaMemcpyHostToDevice)
			);
		}

		virtual void memcpy_to_host()
		{
			//copies the data to cpu
			checkCuda(
				cudaMemcpy((void**)host_array, device_matrix.matirx, VectorDimensions * sizeof(T), cudaMemcpyDeviceToHost)
			);
		}

	private:
		const datastructures::ThreadDevice2DMatrix<T> device_matirx;
	};

}
                