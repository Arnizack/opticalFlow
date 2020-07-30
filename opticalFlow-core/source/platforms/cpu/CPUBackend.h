#pragma once
#include<vector>
#include<array>
#include<functional>

#include"datastructures/Vec.h"
#include"datastructures/Vec2D.h"
#include"datastructures/Vec3D.h"
#include"datastructures/Vec4D.h"
#include"launchers/kernelInfo.h"

#include"schedulers/gridStripScheduler.h"
#include"schedulers/tilesSchedular.h"
#include"platforms/cpu/datastructures/Matrix2D.h"
#include"datastructures/ArrayBuffer.h"
#include"datastructures/MatrixBuffer.h"
#include"platforms/cpu/datastructures/Matrix2D.h"
#include<vector>

namespace cpu
{

	
	
	struct dataStructsCPU
	{
		template<typename T>
		using Array = T*;


		template<class T>
		using Matrix1D = cpu::Matrix2D<T, 1>;



		template<class T>
		using Matrix2D = cpu::Matrix2D<T, 2>;


		template<class T>
		using Matrix3D = cpu::Matrix2D<T, 3>;


		template<class T>
		using Matrix4D = cpu::Matrix2D<T, 4>;


		using TextureRGBA = cpu::Matrix2D<float, 4>;


		using TextureGrayScale = cpu::Matrix2D<float, 1>;



	};

	struct dataTypesCPU
	{

		using int2 = Vec2D<int>;
		using int3 = Vec3D<int>;
		using int4 = Vec4D<int>;

		using float2 = Vec2D<float>;
		using float3 = Vec3D<float>;
		using float4 = Vec4D<float>;

		using kernelInfo = kernelInfo;


		template<typename T>
		using ArrayBuffer = std::vector<T>;

		template<typename T>
		static ArrayBuffer<T> allocArrayBuffer(kernelInfo& kinfo, const int& count)
		{
			return cpu::allocArrayBuffer(count);
		}

		template<typename T>
		using MatrixBuffer = MatrixBuffer<T>;

		template<typename T>
		static MatrixBuffer<T> allocMatrixBuffer(kernelInfo& kinfo, const int& width, const int& heigth)
		{
			return cpu::allocMatrixBuffer(width, heigth,0,0);
		}

		template<typename T>
		using TilesBuffer = MatrixBuffer<T>;

		template<typename T>
		static MatrixBuffer<T> allocTilesBuffer(kernelInfo& kinfo, const Vec2D<int>& dimenions, const Vec2D<int>& tilesSize, const Vec2D<int>& padding)
		{
			return cpu::allocMatrixBuffer<T>(dimenions.x+ 2*padding.x, dimenions.y + 2 * padding.y,-padding.x,-padding.y);
		}

	};


	struct schedularsCPU
	{
		template<typename _inst,typename... ARGS>
		static void gridStripSchedular(kernelInfo info, int itemCount, _inst Instruction, ARGS... args)
		{
	
			cpu::gridStripSchedular(info, itemCount, Instruction, args...);
		}

		template<typename... ARGS, typename _inst>
		inline static void tilesSchedular2D(kernelInfo& info, const Vec2D<int>& dimenions, const Vec2D<int>& tilesSize, const Vec2D<int>& padding,
			_inst instruction, ARGS&&... args)
		{
			cpu::tilesSchedular2D(0, dimenions, tilesSize, padding, instruction, std::forward<ARGS>(args)...);
		}
	};
	
	struct BackendCPU
	{
		//datastructur
		using ds = dataStructsCPU;
		//datatypes
		using dt = dataTypesCPU;

		using sh = schedularsCPU;

	};

}