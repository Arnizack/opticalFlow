#pragma once
#include"Vec.h"

namespace cpu
{

	template<class T>
	struct Vec4D
	{
		T x;
		T y;
		T z;
		T w;

		Vec4D(const Vec<T, 4>& other)
			:x(other[0]), y(other[1]), z(other[2]), w(other[3])
		{}

		Vec4D(const T& X, const T& Y, const T& Z, const T& W)
			:x(X), y(Y), z(Z), w(W)
		{}

		Vec4D() = default;

		inline Vec4D operator+(const Vec4D& a) const
		{
			Vec4D result = *this;
			result += a;
			return result;
		}

		inline void operator+=(const Vec4D& a)
		{
			x += a.x;
			y += a.y;
			z += a.z;
			w += a.w;
		}

		inline Vec4D operator-(const Vec4D& a) const
		{
			Vec4D result = *this;
			result -= a;
			return result;
		}

		inline void operator-=(const Vec4D& a)
		{
			x -= a.x;
			y -= a.y;
			z -= a.z;
			w -= a.w;
		}

		inline Vec4D operator*(const T& s)
		{
			Vec4D result = *this;
			result *= s;
			return s;
		}

		inline void operator*=(const T& s)
		{
			x *= s;
			y *= s;
			z *= s;
			w *= s;
		}

		inline Vec4D operator/(const T& s)
		{
			Vec4D result = *this;
			result /= s;
			return s;
		}

		inline void operator/=(const T& s)
		{
			x /= s;
			y /= s;
			z /= s;
			w /= s;
		}
	};
}
