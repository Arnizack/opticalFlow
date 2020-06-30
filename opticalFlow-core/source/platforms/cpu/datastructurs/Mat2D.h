#pragma once

//#include "DataStructures/Host/host_data.h"
//#include "DataStructures/Host/index.h"
//#include "DataStructures/Host/index2.h"

#include <memory>

namespace cpu
{
	template<typename T, const size_t XDimension, const size_t YDimension>
	struct Mat2D
	{
		const size_t sizeX = XDimension;	//number of elements in the rows
		const size_t sizeY = YDimension;	//number of elements in the columns
		const size_t size;					//number of total elements
		std::unique_ptr<T[]> matrix;		//matrice

		//constructor
		Mat2D();

		//operators:
		//acces
		inline T& operator[](const cpu::index2& i);
		inline const T& operator[](const cpu::index2& i) const;
		T& operator[](const  int& i);
		const T& operator[](const  int& i) const;

		//set
		inline T& operator=(const T& other);

		//adding
		inline void operator+=(const Mat2D& other);
		inline Mat2D& operator+(const Mat2D& other) const;

		//subtracting
		inline void operator-=(const Mat2D& other);
		inline Mat2D& operator-(const Mat2D& other) const;

		//multiplying by value
		inline void operator*=(const T& other);
		inline Mat2D& operator*(const T& other) const;

		//dividing by value
		inline void operator/=(const T& other);
		inline Mat2D& operator/(const T& other) const;
	};
}