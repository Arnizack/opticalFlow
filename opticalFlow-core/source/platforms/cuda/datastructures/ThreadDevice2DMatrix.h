#pragma once
#include <cuda_runtime.h>

namespace datastructures
{
	template<typename T>
	struct ThreadDevice2DMatrix
	{
		T* matrix;
		const size_t width;

		ThreadDevice2DMatrix(const size_t& _width,T* _matrix)
			:width(_width), matrix(_matrix)
		{}


		//Device Operators
		__device__
			T& operator[](const int2& i)
		{
			return matrix[i.x + i.y * width];
		}

		__device__
			const T& operator[](const int2& i) const
		{
			return matrix[i.x + i.y * width];
		}
	};

}
