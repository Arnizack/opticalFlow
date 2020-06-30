#pragma once
#include<array>

namespace cpu
{


	template<typename _Ty,size_t _Size>
	class Vec : public std::array<_Ty, _Size>
	{
	public:

		//using std::array<_Ty, _Size>::array;
		template<typename T>
		Vec operator+(const Vec<T, _Size>& v) const
		{
			Vec<_Ty, _Size> result;
			for (size_t idx = 0; idx < _Size; idx++)
				result[idx] = operator[](idx) + v[idx];
			return result;
		}

		template<typename T>
		Vec operator-(const Vec<T, _Size>& v) const
		{
			Vec<_Ty, _Size> result;
			for (size_t idx = 0; idx < _Size; idx++)
				result[idx] = operator[](idx) - v[idx];
			return result;
		}

		template<typename T>
		Vec operator*(const T& scalar) const
		{
			Vec<_Ty, _Size> result;
			for (size_t idx = 0; idx < _Size; idx++)
				result[idx] = operator[](idx) * scalar;
			return result;
		}

		template<typename T>
		Vec operator/(const T& scalar) const
		{
			Vec<_Ty, _Size> result;
			for (size_t idx = 0; idx < _Size; idx++)
				result[idx] = operator[](idx) / scalar;
			return result;
		}

		template<typename T>
		Vec& operator+=(const Vec<T, _Size>& v)
		{
			for (size_t idx = 0; idx < _Size; idx++)
				operator[](idx) += v[idx];
			return *this;
		}

		template<typename T>
		Vec& operator-=(const Vec<T, _Size>& v)
		{
			for (size_t idx = 0; idx < _Size; idx++)
				operator[](idx) += v[idx];
			return *this;
		}

		template<typename T>
		Vec& operator*=(const T& scalar)
		{
			std::for_each(this->begin(), this->end(), [&](_Ty& cell)
			{
				cell *= scalar;
			});
			return *this;
		}

		template<typename T>
		Vec& operator/=(const T& scalar)
		{
			std::for_each(this->begin(), this->end(), [](_Ty& cell)
			{
				cell /= scalar;
			});
			return *this;
		}

	};
}