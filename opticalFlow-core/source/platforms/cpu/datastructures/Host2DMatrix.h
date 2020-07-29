#pragma once
#include"datastructures/DeviceData.h"
#include "platforms/cpu/datastructures/Matrix2D.h"

#include<vector>
#include<algorithm>

namespace cpu
{
	/*---- nD Vector ----*/
	template<typename T, size_t VectorDimension>
	class Host2DMatrix : public datastructures::IDevice2DMatrix<T, VectorDimension>
	{
	public:
		Host2DMatrix(const cpu::Vec<T, VectorDimension>* const src, const size_t& width, const size_t& height)
			: data(src, width, width*height), datastructures::IDevice2DMatrix<T, VectorDimension>(width, height)
		{}

		Host2DMatrix(const T* const src, const size_t& width, const size_t& height)
			: data(src, width, width* height), datastructures::IDevice2DMatrix<T, VectorDimension>(width, height)
		{}

		void copyTo(T* dst) const
		{
			std::copy_n(data.data(), ItemCount, dst);
		}

		cpu::Matrix2D<T, VectorDimension> get2DMatrix()
		{
			return data;
		}

	private:
		cpu::Matrix2D<T, VectorDimension> data;
	};	
}