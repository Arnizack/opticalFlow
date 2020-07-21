/*#include "Mat.h"

namespace cpu
{
	template<typename T>
	cpu::Mat<T>::Mat(const size_t& count)
		:size(count)
	{
		this->matrix = std::make_unique<T[]>(size);
	}

	template<typename T>
	inline T& Mat<T>::operator[](const cpu::BackendCPU::dt::index2& i)
	{
		//access
		return this->matrix[i];
	}

	template<typename T>
	inline const T& Mat<T>::operator[](const int& i) const
	{
		//acces for const
		return this->matrix[i];
	}

	template<typename T>
	inline T& Mat<T>::operator=(const T & other)
	{
		//set
		this = other;
		return this;
	}

	template<typename T>
	inline void Mat<T>::operator+=(const Mat<T>& other)
	{
		assert(this->size != other.size);

		for (size_t i = 0; i < size; i++)
		{
			this->matrix[i] += other.matrix[i];
		}
	}

	template<typename T>
	inline Mat<T>& Mat<T>::operator+(const Mat<T>& other) const
	{
		Mat<T> result = *this;
		result += other;
		return result;
	}

	template<typename T>
	inline void Mat<T>::operator-=(const Mat<T>& other)
	{
		assert(this->size != other.size);

		for (size_t i = 0; i < size; i++)
		{
			this->matrix[i] -= other.matrix[i];
		}
	}

	template<typename T>
	inline Mat<T>& Mat<T>::operator-(const Mat<T>& other) const
	{
		Mat<T> result = *this;
		result -= other;
		return result;
	}

	template<typename T>
	inline void Mat<T>::operator*=(const T & other)
	{
		assert(this->size != other.size);

		for (size_t i = 0; i < size; i++)
		{
			this->matrix[i] *= other;
		}
	}

	template<typename T>
	inline Mat<T>& Mat<T>::operator*(const T & other) const
	{
		Mat<T> result = *this;
		result *= other;
		return result;
	}

	template<typename T>
	inline void Mat<T>::operator/=(const T & other)
	{
		assert(this->size != other.size);

		for (size_t i = 0; i < size; i++)
		{
			this->matrix[i] /= other.matrix[i];
		}
	}

	template<typename T>
	inline Mat<T>& Mat<T>::operator/(const T & other) const
	{
		Mat<T> result = *this;
		result /= other;
		return result;
	}
}
*/