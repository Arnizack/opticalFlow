#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"platforms/cuda/helper/errorCheckingMacro.cuh"
#include<memory>
#include"platforms/cuda/datastructures/kernelInfo.cuh"
#include<tuple>

template<typename _kernel, typename... ARGS>
__global__ void _main_kernel_shared_memory(ARGS... args)
{

	extern __shared__ int sharedMem[];
	void* start = static_cast<void*>(&sharedMem[0]);

	KernelInfo kinfo(start);
	_kernel::kernel(kinfo, std::forward<ARGS>(args)...);
}

template<typename _kernel, typename... ARGS>
__global__ void _main_kernel(ARGS... args)
{
	KernelInfo kinfo(nullptr);
	_kernel::kernel(kinfo, std::forward<ARGS>(args)...);

}

class KernelLauncher
{
	//see doc/Launch-Parameter-Optimierung
public:
	KernelLauncher(const int& sharedMemoryCount, const int& blockSize);
	//returns true when successfully
	bool considerTilesBuffer(const int& dataWidth, const int& dataHeigth, const int& tilesWidth, const int& tilesHeigth, const int& paddingXRadius, const int& paddingYRadius, const int& typeSize);

	//kernel gets as first Argument KernelInfo
	template<typename _kernel, typename... ARGS>
	void launch(ARGS... args)
	{
		calcParameters();
		if (gridDimX != 0 && gridDimY != 0)
		{
			dim3 gridDim;
			gridDim.x = gridDimX;
			gridDim.y = gridDimY;

			_main_kernel_shared_memory<_kernel, ARGS...> << <gridDim, blockCount, SharedMemoryCount >> > (std::forward<ARGS>(args)...);
		}
		else
		{

			int minGridSize;
			int blockSize;
			cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
				_main_kernel<_kernel, ARGS...>, SharedMemoryCount, 0);
			dim3 gridDim;
			gridDim.x = ceil(sqrt(minGridSize));
			gridDim.y = ceil(sqrt(minGridSize));
			_main_kernel_shared_memory<_kernel, ARGS...> << < gridDim, blockSize, SharedMemoryCount >> > (std::forward<ARGS>(args)...);

		}
	}

	template<typename _kernel, typename... ARGS>
	void launchWithParameters(const int& gridDimX, const int& gridDimY, const int& blockSize, ARGS... args)
	{
		dim3 gridDim;
		gridDim.x = gridDimX;
		gridDim.y = gridDimY;
		_main_kernel<_kernel, ARGS...> << <gridDim, blockSize, SharedMemoryCount >> > (args...);
	}

	void wait();

private:
	void calcParameters();

	double b_0 = 0;
	double b_1 = 0;
	double b_2 = 0;
	double b_3 = 0;

	int SharedMemoryCount = 0;

	int gridDimX = 0;
	int gridDimY = 0;
	int blockCount = 0;
};

