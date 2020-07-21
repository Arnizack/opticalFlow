#pragma once
#include <cuda_runtime.h>

namespace datastructures
{
	template<typename T>
	struct ThreadDevice2DMatrix
	{
		T* matrix = 0;
		const size_t width;

		ThreadDevice2DMatrix(const size_t& _width)
			:width(_width)
		{}

		ThreadDevice2DMatrix(const ThreadDevice2DMatrix& obj) = delete;

		~ThreadDevice2DMatrix()
		{
			cudaFree(matrix);
		}

		//Device Operators
		__device__
			T& operator[](const int2& i)
		{
			matrix[i.x + i.y * width];
		}

		__device__
			const T& operator[](const int2& i) const
		{
			matrix[i.x + i.y * width];
		}
	};

}
