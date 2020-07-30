
#pragma once
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include"platforms/cuda/datastructures/ThreadDevice2DMatrix.h"
#include"platforms/cuda/helper/CudaVecType.h"
#include"platforms/cuda/datastructures/kernelInfo.cuh"
#include"platforms/cuda/datastructures/VectorOperator.cuh"
#include"platforms/cuda/datastructures/tilesBuffer.cuh"
#include"platforms/cuda/datastructures/MatrixBuffer.h"
#include"platforms/cuda/datastructures/ArrayBuffer.h"
#include"platforms/cuda/schedulers/tilesScheduler.cuh"

namespace cuda
{
	
	
	
	struct dataStructsCUDA
	{
		template<typename T>
		using Array = T*;

		
		template<class T>
		using Matrix1D = datastructures::ThreadDevice2DMatrix<T,1>;


		template<class T>
		using Matrix2D = datastructures::ThreadDevice2DMatrix<T,2>;

		

		template<class T>
		using Matrix3D = datastructures::ThreadDevice2DMatrix<T, 3>;


		template<class T>
		using Matrix4D = datastructures::ThreadDevice2DMatrix<T, 4>;


		using TextureRGBA = datastructures::ThreadDevice2DMatrix<float, 4>;


		using TextureGrayScale = datastructures::ThreadDevice2DMatrix<float,1>;



	};

	struct dataTypesCUDA
	{
		
		using int2 = int2;
		using int3 = int3;

		template<typename T>
		using TilesBuffer = tilesBufferRF<T>;

		template<typename T>
		static __device__ TilesBuffer<T> allocTilesBuffer(KernelInfo& kinfo, const int2& dimenions, const int2& tilesSize, const int2& padding)
		{
			return allocTilesBufferRF<T>(kinfo, dimenions, tilesSize, padding);
		}

		template<typename T>
		using ArrayBuffer = T*;

		template<typename T>
		static __device__ ArrayBuffer<T> allocArrayBuffer(KernelInfo& kinfo, const int& count)
		{
			return cuda::allocArrayBufferRF(kinfo, count);
		}

		template<typename T>
		using MatrixBuffer = cuda::MatrixBufferRF<T>;

		template<typename T>
		static __device__ MatrixBuffer<T> allocMatrixBuffer(KernelInfo& kinfo, const int& width, const int& heigth)
		{
			return cuda::allocMatrixBufferRF(kinfo, width, heigth);
		}

		using float2 = float2;
		using float3 = float3;
		using float4 = float4;

		using kernelInfo = KernelInfo;

	};


	struct schedularsCUDA
	{
		template<typename _inst, typename... ARGS>
		static void gridStripSchedular(KernelInfo info, int itemCount, _inst Instruction, ARGS... args)
		{
			
			gridStripSchedular(info, itemCount, Instruction, args...);
		}
	};
	
	struct BackendCUDA
	{
		//datastructur
		using ds = dataStructsCUDA;
		//datatypes
		using dt = dataTypesCUDA;

		using sh = schedularsCUDA;

	};

}