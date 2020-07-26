#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ int2 operator+(const int2& a, const int2& b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}
__device__ int3 operator+(const int3& a, const int3& b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ int4 operator+(const int4& a, const int4& b)
{
	return make_int4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ int2 operator-(const int2& a, const int2& b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}
__device__ int3 operator-(const int3& a, const int3& b)
{
	return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ int4 operator-(const int4& a, const int4& b)
{
	return make_int4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

template<typename T>
__device__ int2 operator*(const int2& a, const T& b)
{
	return make_int2(a.x * b, a.y * b);
}
template<typename T>
__device__ int3 operator*(const int3& a, const T& b)
{
	return make_int3(a.x * b, a.y * b, a.z * b);
}
template<typename T>
__device__ int4 operator*(const int4& a, const T& b)
{
	return make_int4(a.x * b, a.y * b, a.z * b, a.w * b);
}

template<typename T>
__device__ int2 operator/(const int2& a, const T& b)
{
	return make_int2(a.x / b, a.y / b);
}
template<typename T>
__device__ int3 operator/(const int3& a, const T& b)
{
	return make_int3(a.x / b, a.y / b, a.z / b);
}
template<typename T>
__device__ int4 operator/(const int4& a, const T& b)
{
	return make_int4(a.x / b, a.y / b, a.z / b, a.w / b);
}



__device__ float2 operator+(const float2& a, const float2& b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float3 operator+(const float3& a, const float3& b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float4 operator+(const float4& a, const float4& b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ float2 operator-(const float2& a, const float2& b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
__device__ float3 operator-(const float3& a, const float3& b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float4 operator-(const float4& a, const float4& b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

template<typename T>
__device__ float2 operator*(const float2& a, const T& b)
{
	return make_float2(a.x * b, a.y * b);
}
template<typename T>
__device__ float3 operator*(const float3& a, const T& b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}
template<typename T>
__device__ float4 operator*(const float4& a, const T& b)
{
	return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

template<typename T>
__device__ float2 operator/(const float2& a, const T& b)
{
	return make_float2(a.x / b, a.y / b);
}
template<typename T>
__device__ float3 operator/(const float3& a, const T& b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}
template<typename T>
__device__ float4 operator/(const float4& a, const T& b)
{
	return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
