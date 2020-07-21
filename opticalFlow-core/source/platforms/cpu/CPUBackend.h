#pragma once
#include<vector>
#include<array>

#include"datastructures/Mat.h"
#include"datastructures/Vec.h"
#include"datastructures/Vec2D.h"
#include"datastructures/Vec3D.h"
#include"datastructures/Vec4D.h"
#include"datastructures/Mat2D.h"
#include"launchers/kernelInfo.h"
#include<functional>

#include"schedulers/gridStripScheduler.h"

namespace cpu
{

	
	/*
	struct dataStructsCPU
	{
		template<typename T>
		using Array = std::vector<T>;


		template<class T>
		using Matrix1D = Mat<T>;



		template<class T>
		using Matrix2D = Mat<Vec2D<T>>;


		template<class T>
		using Matrix3D = Mat<Vec3D<T>>;


		template<class T>
		using Matrix4D = Mat<Vec4D<T>>;


		using TextureRGBA = Matrix4D<float>;


		using TextureGrayScale = Mat<float>;



	};*/

	struct dataTypesCPU
	{
		using index = int;

		using index2 = Vec2D<int>;
		using index3 = Vec3D<int>;

		using float2 = Vec2D<float>;
		using float3 = Vec3D<float>;
		using float4 = Vec4D<float>;

		using kernelInfo = kernelInfo;

	};


	struct schedularsCPU
	{
		template<typename _inst,typename... ARGS>
		static void gridStripSchedular(kernelInfo info, int itemCount, _inst Instruction, ARGS... args)
		{
	
			cpu::gridStripSchedular(info, itemCount, Instruction, args...);
		}
	};
	/*
	struct BackendCPU
	{
		//datastructur
		using ds = dataStructsCPU;
		//datatypes
		using dt = dataTypesCPU;

		using sh = schedularsCPU;

	};*/

}