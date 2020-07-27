#pragma once
#include"platforms/cuda/schedulers/tilesScheduler.cuh"
#include"kernelInfo.cuh"

template<class T>
struct tilesBufferRF
{
	T* Address;
	const int Width;
	const int2 Offset;
	inline __device__ tilesBufferRF(T* dst, const int& width, const int2& offset) : Width(width), Offset(offset), Address(dst)
	{}



	inline __device__ T& operator[](const int2& idx)
	{
		int real_idx = idx.x - Offset.x + (idx.y - Offset.y) * Width;

		return static_cast<T>(Address[real_idx]);
	}
};

template<class T>
struct tilesBufferCF
{
	T* Address;
	const int Heigth;
	const int2 Offset;
	inline __device__ tilesBufferCF(T* dst, const int& heigth, const int2& offset) : Heigth(heigth), Offset(offset), Address(dst)
	{}



	inline __device__ T& operator[](const int2& idx)
	{
		return static_cast<T>(Address[(idx.x - Offset.x) * Heigth + idx.y - Offset.y]);
	}
};

template<class T>
inline __device__ tilesBufferRF<T> allocTilesBufferRF(KernelInfo& kInfo, const int2& dimenions, const int2& tilesSize, const int2& padding)
{
	int2 min;
	int2 max;
	_calcTilesBlockSize(std::forward<const int2&>(dimenions), std::forward<const int2&>(tilesSize),
		std::forward<const int2&>(padding), min, max);

	int size = (max.x - min.x) * (max.y - min.y);




	T* bufferStart = (T*)kInfo.SharedMemStart;
	const auto result = tilesBufferRF<T>(bufferStart, max.x - min.x, min);

	int memOffset = sizeof(T) * size;// (max.x - min.x)* (max.y - min.y);

	kInfo.SharedMemStart = &kInfo.SharedMemStart[memOffset];
	return result;
}