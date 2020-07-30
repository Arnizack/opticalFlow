#pragma once
#include"Vec2D.h"
#include<memory>
#include<vector>

namespace cpu
{
	template<class T>
	struct MatrixBuffer
	{
		MatrixBuffer(const int& width,const int& heigth,const int& offsetX,const int& offsetY): Width(width), Data(width*heigth)
		{}
		T& operator[](const Vec2D<T>& idx)
		{
			return Data[(idx.y-OffsetY) * Width + (idx.x - OffsetX)];
		}
		T operator[](const Vec2D<T>& idx) const
		{
			return Data[(idx.y - OffsetY) * Width + (idx.x - OffsetX)];
		}

	private:
		const int Width;
		const int OffsetX;
		const int OffsetY;
		std::vector<T> Data;
	};

	template<class T>
	MatrixBuffer<T> allocMatrixBuffer(const int& width, const int& heigth, const int& offsetX, const int& offsetY)
	{
		return MatrixBuffer(width, heigth,offsetX,offsetY);
	}
}