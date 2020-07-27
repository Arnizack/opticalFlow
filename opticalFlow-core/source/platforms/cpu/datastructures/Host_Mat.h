#pragma once
#include"datastructures/DeviceData.h"
#include"platforms/cpu/CPUBackend.h"

#include<vector>
#include<algorithm>

namespace cpu
{
	template<typename T, size_t VectorDimension>
	class Host_Mat : public datastructures::IDevice2DMatrix<T, VectorDimension>
	{
	public:
		Host_Mat(cpu::Vec<T, VectorDimension>* const src, const size_t& width, const size_t& height)
			: datastructures::IDevice2DMatrix(width, height), data(width*height)
		{
			std::copy_n(src, ItemCount, data.data())
		}

	private:
		std::vector <cpu::Vec<T, VectorDimension>> data;
	};

	template<typename T, 0>
	class Host_Mat : public datastructures::IDevice2DMatrix<T, 0>
	{};

	/*
	template<class T>
	void __setVector(Vec2D<T>& vec, T* src)
	{
		vec.x = src[0];
		vec.y = src[1];
	}

	template<class T>
	void __setVector(Vec3D<T>& vec, T* src)
	{
		vec.x = src[0];
		vec.y = src[1];
		vec.z = src[2];
	}

	template<class T>
	void __setVector(Vec4D<T>& vec, T* src)
	{
		vec.x = src[0];
		vec.y = src[1];
		vec.z = src[2];
		vec.w = src[3];
	}

	template<class T>
	void __setVector(T& vec, T* src)
	{
		 vec = src[0];
	}

	template<class T>
	void __getVector(Vec2D<T>& vec, T* src)
	{
		 src[0] = vec.x;
		 src[1] = vec.y;
	}

	template<class T>
	void __getVector(Vec3D<T>& vec, T* src)
	{
		src[0]	= vec.x;
		src[1] = vec.y;
		src[2] = vec.z;
	}			

	template<class T>
	void __getVector(Vec4D<T>& vec, T* src)
	{
		src[0] = vec.x;
		src[1] = vec.y;
		src[2] = vec.z;
		src[3] = vec.w;
	}

	template<class T>
	void __getVector(T& vec, T* src)
	{
		 src[0]= vec;
	}


	template<class T,class vecT,int _dim>
	class Host_Mat : public datastructures::IDevice2DMatrix<vecT, _dim>
	{
		using Matrix2D = Mat<T>;
		//using index2 = cpu::BackendCPU::dt::index2;
		
	private:
		std::unique_ptr<Matrix2D> data = nullptr;
	public:
		Host_Mat(vecT* srcData,int width, int heigth)
			: datastructures::IDevice2DMatrix<vecT, _dim>(width,heigth)
		{
			index2 dim;
			dim.x = width;
			dim.y = heigth;

			

			data = std::make_unique<Matrix2D>(dim);

			for (int i = 0; i < width * heigth ; i++)
			{
				
				auto& item = data->operator[](i);
				__setVector<vecT>(item, srcData + i * _dim);
			}

		}

		void copyTo(vecT* dst) const
		{
			for (int i = 0; i < Width * Heigth; i++)
			{
				auto& item = data->operator[](i);
				__getVector<vecT>(item, dst + i * _dim);
			}
		}
	};
	*/
	
}