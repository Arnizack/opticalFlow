#pragma once

namespace cpu
{

	template<class T>
	struct Vec3D
	{
		T x=0;
		T y=0;
		T z=0;

		Vec3D()
		{

		}

		Vec3D(const Vec<T, 3>& other)
		{
			x = other[0];
			y = other[1];
			z = other[2];
		}


		inline Vec3D operator+(const Vec2D& a) const
		{
			Vec3D result = *this;
			result += a;
			return result;
		}

		inline void operator+=(const Vec2D& a)
		{
			x += a.x;
			y += a.y;
			z += a.z;
		}

		inline Vec3D operator-(const Vec2D& a) const
		{
			Vec3D result = *this;
			result -= a;
			return result;
		}

		inline void operator-=(const Vec2D& a)
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