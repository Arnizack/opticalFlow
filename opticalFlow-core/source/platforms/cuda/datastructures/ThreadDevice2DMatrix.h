#pragma once
#include <cuda_runtime.h>
#include"platforms/cuda/helper/CudaVecType.h"
namespace datastructures
{
	template<typename T, int dim>
	struct ThreadDevice2DMatrix
	{

		typedef typename cuda_help::template get_cuda_matrix_type<T, dim>::type vecType;


		vecType* matrix;
		const size_t width;

		ThreadDevice2DMatrix(const size_t& _width,T* _matrix)
			:width(_width), matrix((vecType*)_matrix)
		{}


		//Device Operators
		__device__
			vecType& operator[](const int2& i)
		{
			return matrix[i.x + i.y * width];
		}

		__device__
			const vecType& operator[](const int2& i) const
		{
			return matrix[i.x + i.y * width];
		}
	};

}
