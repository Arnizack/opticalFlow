/*#include "Mat2D.h"

namespace cpu
{
	//constructor
	template<typename T>
	cpu::Mat2D<T>::Mat2D(const size_t& xDimension, const size_t& yDimension)
		:size(xDimension* yDimension), XDimension(xDimension) , YDimension(yDimension)
	{
		this->matrix = std::make_unique<T[]>(size);
	}

	//acces
	template<typename T, const size_t X, const size_t Y>
	inline T& cpu::Mat2D<T, X, Y>::operator[](const cpu::index2& i)
	{
		return this->matrix.get()[i.x + sizeX * i.y];
	}

	template<typename T, const size_t X, const size_t Y>
	inline const T& cpu::Mat2D<T, X, Y>::operator[](const cpu::index2& i) const
	{
		return this->matrix.get()[i.x + sizeX * i.y];
	}
	
	template<typename T, const size_t X, const size_t Y>
	T & cpu::Mat2D<T, X, Y>::operator[](const int& i)
	{
		return this->matrix.get()[i];
	}

	template<typename T, const size_t X, const size_t Y>
	const T & cpu::Mat2D<T, X, Y>::operator[](const int& i) const
	{
		return this->matrix.get()[i];
	}

	//set
	template<typename T, const size_t X, const size_t Y>
	T & cpu::Mat2D<T, X, Y>::operator=(const T & other)
	{
		this = other;
		return this;
	}

	//add
	template<typename T, const size_t X, const size_t Y>
	inline void cpu::Mat2D<T, X, Y>::operator+=(const Mat2D<T, X, Y>& other)
	{
		for (size_t i = 0; i < size; i++)
		{
			this->matrix[i] += other.matrix[i];
		}
	}

	template<typename T, const size_t X, const size_t Y>
	inline cpu::Mat2D<T, X, Y> & Mat2D<T, X, Y>::operator+(const Mat2D<T, X, Y>& other) const
	{
		Mat2D<T> result = *this;
		result += other;
		return result;
	}

	//sub
	template<typename T, const size_t X, const size_t Y>
	inline void cpu::Mat2D<T, X, Y>::operator-=(const Mat2D<T, X, Y>& other)
	{
		for (size_t i = 0; i < size; i++)
		{
			this->matrix[i] -= other.matrix[i];
		}
	}

	template<typename T, const size_t X, const size_t Y>
	inline cpu::Mat2D<T, X, Y> & Mat2D<T, X, Y>::operator-(const Mat2D<T, X, Y>& other) const
	{
		Mat2D<T> result = *this;
		result -= other;
		return result;
	}

	//mult
	template<typename T, const size_t X, const size_t Y>
	inline void cpu::Mat2D<T, X, Y>::operator*=(const T & other)
	{
		for (size_t i = 0; i < size; i++)
		{
			this->matrix[i] *= other.matrix[i];
		}
	}

	template<typename T, const size_t X, const size_t Y>
	inline cpu::Mat2D<T, X, Y> & Mat2D<T, X, Y>::operator*(const T & other) const
	{
		Mat2D<T> result = *this;
		result *= other;
		return result;
	}

	//div
	template<typename T, const size_t X, const size_t Y>
	inline void cpu::Mat2D<T, X, Y>::operator/=(const T & other)
	{
		for (size_t i = 0; i < size; i++)
		{
			this->matrix[i] /= other.matrix[i];
		}
	}

	template<typename T, const size_t X, const size_t Y>
	inline cpu::Mat2D<T, X, Y>& Mat2D<T, X, Y>::operator/(const T & other) const
	{
		Mat2D<T> result = *this;
		result /= other;
		return result;
	}

}
*/