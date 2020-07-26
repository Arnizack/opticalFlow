#include "CUDADeviceBuilder.h"
#include "cuda_runtime.h"
#include "platforms/cuda/CUDADataStructsFactory.hpp"
#include "platforms/cuda/CUDADeviceMemAccess.hpp"
#include"CUDAKernels.h"

bool kernel::CUDADeviceBuilder::isAvailable()
{
	int count;
	cudaGetDevice(&count);
	if (count > 0)
		return true;
	return false;
}

std::unique_ptr<kernel::IKernels> kernel::CUDADeviceBuilder::createKernels()
{
	return std::make_unique<kernel::CUDAKernels>();
}

std::unique_ptr<kernel::IDataStructuresFactory> kernel::CUDADeviceBuilder::createDataStrucFactory()
{
	auto result = std::make_unique<cuda::CUDADataStructsFactory>();
	return result;
}

std::unique_ptr<kernel::IDeviceMemAccess> kernel::CUDADeviceBuilder::createMemAccesser()
{
	auto result = std::make_unique<cuda::CUDADeviceMemAccess>();
	return result;
}
