#pragma once

#include"datastructures/DeviceData.h"
#include "HostDeviceObj.h"
#include "ThreadDevice2DMatrix.h"
#include"platforms/cuda/helper/CudaVecType.h"

namespace datastructures
{

	

	template<typename T, size_t VectorDimensions>
	class HostDevice2DMatrix : public datastructures::IDevice2DMatrix<T, VectorDimensions>, public HostDeviceObj
	{
	public:
		
		//using internCudaVecType = cuda_help::template get_cuda_matrix_type<T, VectorDimensions>::type;

		HostDevice2DMatrix(T* const& matrix, const size_t& width, const size_t& height)
			: IDevice2DMatrix( width, height )
		{
			allocate_gpu();
			
			memcpy_to_device(matrix);
		}

		virtual ~HostDevice2DMatrix() override
		{
			cudaFree(device_matrix);
		}

		constexpr size_t size() noexcept
		{
			return Width * Heigth * VectorDimensions;
		}

		datastructures::ThreadDevice2DMatrix<T, VectorDimensions> getCuda2DMatrix()
		{
			//could return not initialized Data
			return datastructures::ThreadDevice2DMatrix<T, VectorDimensions>(Width, device_matrix);
		}

	protected:
		virtual void allocate_gpu()
		{
			//allocates Memory on the gpu for the matrix
			checkCuda(
				cudaMalloc(&device_matrix, Width*Heigth* VectorDimensions * sizeof(T)) // vlt (void**)
			);
		}

		virtual void memcpy_to_device(void* const& src) override
		{
			//copies the data to gpu
			checkCuda(
				cudaMemcpy(device_matrix, src, Width * Heigth * VectorDimensions * sizeof(T), cudaMemcpyHostToDevice)
			);
		}

		virtual void memcpy_to_host(void* dst) const override
		{
			//copies the data to cpu
			checkCuda(
				cudaMemcpy(dst, device_matrix, Width * Heigth * VectorDimensions * sizeof(T), cudaMemcpyDeviceToHost)
			);
		}

	private:
		T* device_matrix;
	};

}
                