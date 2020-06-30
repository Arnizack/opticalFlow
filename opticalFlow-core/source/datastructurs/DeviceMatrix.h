#pragma once
namespace datastructures
{
	template<typename T, size_t VectorDimensions>
	class Device2DMatrix
	{
		Device2DMatrix(const size_t& count, const T *const data);
		~Device2DMatrix();
		
		//Device and Host Operators
		__device__ 
			T& operator[](const int2& i);
		__device__
			const T& operator[](const int2& i) const;

		const size_t Width;
		const size_t Heigth;

	protected:
		size_t size;
		T *host_data;
		T *device_data = nullptr;
		inline cudaError_t checkCuda(cudaError_t result);
	};

}
                