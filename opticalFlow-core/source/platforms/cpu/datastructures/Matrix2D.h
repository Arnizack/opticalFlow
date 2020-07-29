#pragma once
#include <vector>
#include <algorithm>
#include "platforms/cpu/CPUBackend.h"


namespace cpu
{
	/*---- nDim Vector ----*/
	template<typename T, size_t VectorDim>
	struct Matrix2D
	{
		std::vector<cpu::Vec<T, VectorDim>> matrix;
		const size_t width;

		//constructor
		Matrix2D(const cpu::Vec<T, VectorDim> *const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			std::copy_n(src, ItemCount, matix.data());
		}

		Matrix2D(const T* const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i += VectorDim)
			{
				for (auto j = 0; j < VectorDim; j++)
				{
					matrix[i][j] = src[i + j];
				}
			}
		}

		//acces
		inline cpu::Vec<T, VectorDim>& operator[](const cpu::dataTypesCPU::index2& i)
		{
			return data[i.x + width * i.y];
		}

		inline const cpu::Vec<T, VectorDim>& operator[](const cpu::dataTypesCPU::index2& i) const
		{
			return data[i.x + width * i.y];
		}

		cpu::Vec<T, VectorDim>* data() noexcept
		{
			return matrix.data();
		}

		const cpu::Vec<T, VectorDim>* data() const noexcept
		{
			return matrix.data();
		}

	};

	/*---- 1Dim Vector ----*/
	template<typename T>
	struct Matrix2D<T, 1>
	{
		std::vector<T> matrix;
		const size_t width;

		//constructor
		Matrix2D(const cpu::Vec<T, 1>* const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i++)
			{
				matrix[i] = src[i][0];
			}
		}

		Matrix2D(const T* const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i++)
			{
				matrix[i] = src[i];
			}
		}

		//acces
		inline T& operator[](const cpu::dataTypesCPU::index2& i)
		{
			return data[i.x + width * i.y];
		}

		inline const T& operator[](const cpu::dataTypesCPU::index2& i) const
		{
			return data[i.x + width * i.y];
		}

		T* data() noexcept
		{
			return matrix.data();
		}

		const T* data() const noexcept
		{
			return matrix.data();
		}
	};

	/*---- 2Dim Vector ----*/
	template<typename T>
	struct Matrix2D<T, 2>
	{
		std::vector<cpu::Vec2D<T>> matrix;
		const size_t width;

		//constructor
		Matrix2D(const cpu::Vec<T, 2>* const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i++)
			{
				matrix[i] = cpu::Vec2D<T>(src[i]);
			}
		}

		Matrix2D(const T* const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i += 2)
			{
				matrix[i] = cpu::Vec2D<T>(src[i], src[i+1]);
			}
		}

		//acces
		inline cpu::Vec2D<T>& operator[](const cpu::dataTypesCPU::index2& i)
		{
			//int pos = i.x + (width * i.y);
			return this->matrix[i.x + (width * i.y)];
		}

		inline const cpu::Vec2D<T>& operator[](const cpu::dataTypesCPU::index2& i) const
		{
			//int pos = i.x + (width * i.y);
			return this->matrix[i.x + (width * i.y)];
		}

		cpu::Vec2D<T>* data() noexcept
		{
			return matrix.data();
		}

		const cpu::Vec2D<T>* data() const noexcept
		{
			return matrix.data();
		}
	};

	/*---- 3Dim Vector ----*/
	template<typename T>
	struct Matrix2D<T, 3>
	{
		std::vector<cpu::Vec3D<T>> matrix;
		const size_t width;

		//constructor
		Matrix2D(const cpu::Vec<T, 3>* const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i++)
			{
				matrix[i] = cpu::Vec3D<T>(src[i]);
			}
		}

		Matrix2D(const T* const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i += 3)
			{
				matrix[i] = cpu::Vec2D<T>(src[i], src[i + 1], src[i + 2]);
			}
		}

		//acces
		inline cpu::Vec3D<T>& operator[](const cpu::dataTypesCPU::index2& i)
		{
			return data[i.x + width * i.y];
		}

		inline const cpu::Vec3D<T>& operator[](const cpu::dataTypesCPU::index2& i) const
		{
			return data[i.x + width * i.y];
		}

		cpu::Vec3D<T>* data() noexcept
		{
			return matrix.data();
		}

		const cpu::Vec3D<T>* data() const noexcept
		{
			return matrix.data();
		}
	};

	/*---- 4Dim Vector ----*/
	template<typename T>
	struct Matrix2D<T, 4>
	{
		std::vector<cpu::Vec4D<T>> matrix;
		const size_t width;

		//constructor
		Matrix2D(const cpu::Vec<T, 4>* const src, const size_t& Width, const size_t& Height)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i++)
			{
				matrix[i] = cpu::Vec4D<T>(src[i]);
			}
		}

		Matrix2D(const T* const src, const size_t& Width, const size_t& ItemCount)
			: matrix(ItemCount), width(Width)
		{
			for (auto i = 0; i < ItemCount; i += 4)
			{
				matrix[i] = cpu::Vec2D<T>(src[i], src[i + 1], src[i + 2], src[i + 3]);
			}
		}

		//acces
		inline cpu::Vec4D<T>& operator[](const cpu::dataTypesCPU::index2& i)
		{
			return data[i.x + width * i.y];
		}

		inline const cpu::Vec4D<T>& operator[](const cpu::dataTypesCPU::index2& i) const
		{
			return data[i.x + width * i.y];
		}

		cpu::Vec4D<T>* data() noexcept
		{
			return matrix.data();
		}

		const cpu::Vec4D<T>* data() const noexcept
		{
			return matrix.data();
		}
	};
}