#pragma once
#include<vector>
#include<array>

#include"datastructurs/Mat.h"
#include"datastructurs/Vec.h"
#include"datastructurs/Vec2D.h"
#include"datastructurs/Vec3D.h"
#include"datastructurs/Vec4D.h"

namespace cpu
{

	struct BackendCPU
	{
		//datastructur
		using ds = dataStructsCPU;
		//datatypes
		using dt = dataTypesCPU;

	};

	struct dataStructsCPU
	{
		template<typename T>
		using Array = std::vector;

		using Matrix1D = Mat<float>;

		template<class T>
		using Matrix2D = Mat<Vec2D<T>>;
		template<class T>
		using Matrix3D = Mat<Vec3D<T>>;
		template<class T>
		using Matrix4D = Mat<Vec4D<T>>;


		using TextureRGBA = Matrix4D<float>;

		using TextureGrayScale = Mat<float>;


	};

	struct dataTypesCPU
	{
		using index = int;

		using index2 = Vec2D<int>;
		using index3 = Vec3D<int>;

		using float2 = Vec2D<float>;
		using float3 = Vec3D<float>;
		using float4 = Vec4D<float>;

	};

}