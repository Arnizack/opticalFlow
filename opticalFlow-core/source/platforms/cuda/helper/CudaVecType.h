#pragma once
#include <cuda_runtime.h>

namespace cuda_help
{
	template<typename T, size_t VectorDimensions>
	struct get_cuda_matrix_type
	{
		using type = T[VectorDimensions];
	};

	template<>
	struct get_cuda_matrix_type<float, 1>
	{
		using type = float;
	};
	template<>
	struct get_cuda_matrix_type<float, 2>
	{
		using type = float2;
	};
	template<>
	struct get_cuda_matrix_type<float, 3>
	{
		using type = float3;
	};
	template<>
	struct get_cuda_matrix_type<float, 4>
	{
		using type = float4;
	};
	template<>
	struct get_cuda_matrix_type<int, 1>
	{
		using type = int;
	};
	template<>
	struct get_cuda_matrix_type<int, 2>
	{
		using type = int2;
	};
	template<>
	struct get_cuda_matrix_type<int, 3>
	{
		using type = int3;
	};
	template<>
	struct get_cuda_matrix_type<int, 4>
	{
		using type = int4;
	};
}