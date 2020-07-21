#pragma once
#include <memory>
#include <cassert>
/*#include"Mat.h"

namespace cpu
{
	template<typename T>
	struct Mat
	{
		std::unique_ptr<T> matrix;		//matrice
		const size_t size;				//size of matrix

		//constructor
		Mat(const size_t& count);

		//operators:
		//acces
		inline T& operator[](const cpu::BackendCPU::dt::index2& i);
		inline const T& operator[](const cpu::BackendCPU::dt::index2& i) const;

		//set
		inline T& operator=(const T& other);

		//adding
		inline void operator+=(const Mat<T>& other);
		inline Mat& operator+(const Mat<T>& other) const;

		//subtracting
		inline void operator-=(const Mat<T>& other);
		inline Mat& operator-(const Mat<T>& other) const;

		//multiplying by value
		inline void operator*=(const T& other);
		inline Mat& operator*(const T& other) const;

		//dividing by value
		inline void operator/=(const T& other);
		inline Mat& operator/(const T& other) const;
	};
}*/