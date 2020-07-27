
#pragma once
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include"platforms/cuda/datastructures/ThreadDevice2DMatrix.h"
#include"platforms/cuda/helper/CudaVecType.h"
#include"platforms/cuda/datastructures/kernelInfo.cuh"
#include"platforms/cuda/datastructures/VectorOperator.cuh"

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
			
			cpu::gridStripSchedular(info, itemCount, Instruction, args...);
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