#include "../../../opticalFlow-core/source/platforms/cuda/datastructures/HostDeviceArray.h"

#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ 
void kernelAusprobieren(datastructures::ThreadDeviceArray<int> temp, int length)
{
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		for (int i = 0; i < length; i++)
		{
			printf("%d", temp[i]);
		}
	}
}

__global__ void globatest () 
{}

int main()
{
	int size = 2;
	int arr[2] = { 1,1 };
	int* ptr = arr;

	datastructures::HostDeviceArray<int> tempObj(ptr, 2);

	std::cout << tempObj.size() << '\n';

	datastructures::ThreadDeviceArray<int> testThreadArray = tempObj.getCudaArray();

	dim3 grid = 1;

	kernelAusprobieren<<<grid, 1 >>>(testThreadArray, size);

	return 0;
}