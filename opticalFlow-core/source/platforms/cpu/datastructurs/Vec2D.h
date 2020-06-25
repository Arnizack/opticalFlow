#pragma once
#include"Vec.h"

namespace cpu
{

	template<class T>
	struct Vec2D
	{
		T x;
		T y;

		Vec2D(const Vec<T, 2>& other)
		{
			x = other[0];
			y = other[1];
		}

		inline Vec2D operator+(const Vec2D& a) const
		{
			Vec2D result = *this:
			result += a;
			return result;
		}

		inline void operator+=(const Vec2D& a)
		{
			x += a.x;
			y += a.y;
		}

		inline Vec2D operator-(const Vec2D& a) const
		{
			Vec2D result = *this:
			result -= a;
			return result;
		}

		inline void operator-=(const Vec2D& a)
		{
			x -= a.x;
			y -= a.y;
		}

		inline Vec2D operator*(const T& s)
		{
			Vec2D result = *this:
			result *= s;
			return s;
		}

		inline void operator*=(const T& s)
		{
			x *= s;
			y *= s;

		}

		inline Vec2D operator/(const T& s)
		{
			Vec2D result = *this:
			result /= s;
			return s;
		}

		inline void operator/=(const T& s)
		{
			x /= s;
			y /= s;

		}
	};
}