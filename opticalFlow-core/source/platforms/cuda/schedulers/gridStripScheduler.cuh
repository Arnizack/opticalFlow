#include"platforms/cuda/datastructures/kernelInfo.cuh"
#include<cuda_runtime.h>
#include<device_launch_parameters.h>

namespace cuda
{

	template<typename _inst, typename... ARGS>
	void gridStripSchedular(KernelInfo info, int itemCount, _inst Instruction, ARGS... args)
	{
		for (int index = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * blockDim.x +
			blockIdx.y * blockDim.x + blockIdx.x;
			index < itemCount;
			index += gridDim.x * blockIdx.x + gridDim.y * blockDim.x * blockDim.y
			)
		{
			Instruction(index, std::forwad<ARGS>(args)...);
		}
	}
}