#pragma once

#include <vector>
#include <memory>


namespace cpu
{
	template<class T, size_t S>
	struct Vec
	{
		std::unique_ptr<T[]> data;
		const size_t length;

		Vec()
			:length(S)
		{
			this->data = std::make_unique<T[]>(this->length);
		}

		inline T& operator[](const int& i)
		{
			return this->data.get()[i];
		}

		inline const T& operator[] (const int& i) const
		{
			return this->data.get()[i];
		}

		inline T& operator=(const T& other)
		{
			this = other;
			return *this;
		}

		inline Vec operator+(const Vec& a) const
		{
			Vec result = *this;
			result += a;
			return result;
		}

		inline void operator+=(const Vec& a)
		{
			for (size_t i = 0; i < length; i++)
			{
				data[i] += a[i];
			}
		}

		inline Vec operator-(const Vec& a) const
		{
			Vec result = *this;
			result -= a;
			return result;
		}

		inline void operator-=(const Vec& a)
		{
			for (size_t i = 0; i < length; i++)
			{
				data[i] -= a[i];
			}
		}

		inline Vec operator*(const T& s)
		{
			Vec result = *this;
			result *= s;
			return s;
		}

		inline void operator*=(const T& s)
		{
			for (size_t i = 0; i < length; i++)
			{
				data[i] *= s;
			}
		}

		inline Vec operator/(const T& s)
		{
			Vec result = *this;
			result /= s;
			return s;
		}

		inline void operator/=(const T& s)
		{
			for (size_t i = 0; i < length; i++)
			{
				data[i] /= s;
			}
		}
	};
}