#pragma once
#include"Vec.h"

namespace cpu
{

	template<class T>
	struct Vec3D
	{
		T x;
		T y;
		T z;

		Vec3D(const Vec<T, 3>& other)
			: x(other[0]), y(other[1]), z(other[2])
		{}

		Vec3D(const T& X, const T& Y, const T& Z)
			:x(X), y(Y), z(Z)
		{}

		Vec3D() = default;

		inline Vec3D operator+(const Vec3D& a) const
		{
			Vec3D result = *this;
			result += a;
			return result;
		}

		inline void operator+=(const Vec3D& a)
		{
			x += a.x;
			y += a.y;
			z += a.z;
		}

		inline Vec3D operator-(const Vec3D& a) const
		{
			Vec3D result = *this;
			result -= a;
			return result;
		}

		inline void operator-=(const Vec3D& a)
		{
			x -= a.x;
			y -= a.y;
			z -= a.z;
		}

		inline Vec3D operator*(const T& s)
		{
			Vec3D result = *this;
			result *= s;
			return s;
		}

		inline void operator*=(const T& s)
		{
			x *= s;
			y *= s;
			z *= s;

		}

		inline Vec3D operator/(const T& s)
		{
			Vec3D result = *this;
			result /= s;
			return s;
		}

		inline void operator/=(const T& s)
		{
			x /= s;
			y /= s;
			z /= s;
		}
	};

}